import random
from threading import RLock
from typing import List, Tuple, Dict, Optional

# 请根据你的实际路径调整导入
from ...base import Function


class ExternalArchive:
    """
    ExternalArchive (Global External Memory Pool)

    Acts at the ClusterManager level to record the global evolutionary trajectory.
    Implements a 'Waterfall' and 'Hard Negatives' mechanism:
    1. Elites: Maintains strict purity, keeping only the Global Top-K candidates.
    2. Hard Negatives: High-scoring candidates ousted from the Elite pool are demoted here.
    3. SE Context: Provides cached, high-contrast context (Elites vs. Hard Negatives)
       for the Semantic Exploration (SE) operator, fully thread-safe with fault tolerance.
    """

    def __init__(self, max_elites: int = 5, max_hard_negatives: int = 30, summary_update_interval: int = 12):
        self.max_elites = max_elites
        self.max_hard_negatives = max_hard_negatives

        # Descending order Top Tier
        self.elites: List[Function] = []
        # Descending order Second Tier (Strongest failures / ousted elites)
        self.hard_negatives: List[Function] = []

        self._lock = RLock()

        # --- SE Operator Caching & Concurrency Control ---
        self._cached_summary: str = ""
        self._request_counter: int = 0
        self._summary_update_interval: int = summary_update_interval

        # Dirty flag: True if new valuable candidates have been added since the last summary
        self._has_unsummarized_updates: bool = False
        # Concurrency Lock: True if a thread is currently calling the LLM to generate a summary
        self._is_summary_generation_locked: bool = False

    def register(self, candidate: Function):
        """Main entry point: evaluate and register a new candidate function."""
        if candidate.score is None:
            return

        with self._lock:
            target_code = candidate.body

            # === 1. Global Deduplication & Dynamic Promotion/Demotion ===

            # Check Elites
            for i, elite in enumerate(self.elites):
                if elite.body == target_code:
                    if candidate.score > elite.score:
                        self.elites[i] = candidate
                        self.elites.sort(key=lambda x: x.score, reverse=True)
                        self._has_unsummarized_updates = True
                    return

                    # Check Hard Negatives
            for i, hard_negative in enumerate(self.hard_negatives):
                if hard_negative.body == target_code:
                    if candidate.score > hard_negative.score:
                        self.hard_negatives[i] = candidate
                        self._has_unsummarized_updates = True

                        # [Dynamic Promotion] Try to push the improved candidate into Elites
                        upgraded_candidate = self.hard_negatives.pop(i)
                        self._try_add_to_elites(upgraded_candidate)
                    return

                    # === 2. New Code: Execute Waterfall insertion ===
            self._try_add_to_elites(candidate)

    def _try_add_to_elites(self, candidate: Function):
        """Attempts to add to Elites; demotes ousted candidates to Hard Negatives."""
        if len(self.elites) < self.max_elites:
            self.elites.append(candidate)
            self.elites.sort(key=lambda x: x.score, reverse=True)
            self._has_unsummarized_updates = True
        else:
            if candidate.score > self.elites[-1].score:
                # Insert and pop the weakest elite
                self.elites.append(candidate)
                self.elites.sort(key=lambda x: x.score, reverse=True)
                ousted_elite = self.elites.pop()
                self._has_unsummarized_updates = True

                # Waterfall demotion: ousted elite goes to hard negatives
                self._try_add_to_hard_negatives(ousted_elite)
            else:
                # Not strong enough for elites, try hard negatives
                self._try_add_to_hard_negatives(candidate)

    def _try_add_to_hard_negatives(self, candidate: Function):
        """Maintains the secondary pool of high-quality failed attempts."""
        if len(self.hard_negatives) < self.max_hard_negatives:
            self.hard_negatives.append(candidate)
            self.hard_negatives.sort(key=lambda x: x.score, reverse=True)
            self._has_unsummarized_updates = True
        else:
            if candidate.score > self.hard_negatives[-1].score:
                self.hard_negatives.append(candidate)
                self.hard_negatives.sort(key=lambda x: x.score, reverse=True)
                self.hard_negatives.pop()  # Discard the absolute weakest
                self._has_unsummarized_updates = True

    def _sample_for_contrastive_context(self, k_elites: int = 4, k_negatives: int = 4) -> Tuple[
        List[Function], List[Function]]:
        """Samples individuals for the SE operator."""
        with self._lock:
            k_e = min(len(self.elites), k_elites)
            sampled_elites = self.elites[:k_e]

            k_n = min(len(self.hard_negatives), k_negatives)
            sampled_negatives = random.sample(self.hard_negatives, k_n) if k_n > 0 else []

            return sampled_elites, sampled_negatives

    # ==========================================
    # Multi-threading Dispatch Interfaces
    # ==========================================

    def fetch_summary_context(self) -> Tuple[str, bool, Dict[str, List[Function]]]:
        """
        Called by the outer 'se' operator.
        Returns: (current_cached_summary, requires_update_flag, context_samples_dict)
        """
        with self._lock:
            self._request_counter += 1

            # Prevent hallucination if archive is entirely empty
            if not self.elites and not self.hard_negatives:
                return self._cached_summary, False, {}

            is_cycle_reached = (self._request_counter % self._summary_update_interval == 0)

            requires_new = (
                    not self._cached_summary
                    or (is_cycle_reached and self._has_unsummarized_updates)
            )

            # Check if update is needed AND no other thread is currently updating
            if requires_new and not self._is_summary_generation_locked:
                self._is_summary_generation_locked = True  # Acquire virtual lock
                elite_samples, negative_samples = self._sample_for_contrastive_context()
                context_dict = {'elites': elite_samples, 'hard_negatives': negative_samples}
                return self._cached_summary, True, context_dict
            else:
                # Return cached data if no update needed, or if another thread is already generating
                return self._cached_summary, False, {}

    def update_global_summary(self, generated_summary: Optional[str]):
        """
        Called by the LLM thread to write back the generated summary and release lock.
        Handles failed LLM generations gracefully.
        """
        with self._lock:
            # 1. Unconditionally release the generation lock to prevent deadlocks
            self._is_summary_generation_locked = False

            # 2. Validate the generation
            if generated_summary and generated_summary.strip():
                self._cached_summary = generated_summary
                # Reset dirty flag only upon successful generation
                self._has_unsummarized_updates = False
            else:
                # On failure: Do nothing. The dirty flag remains True,
                # so the next thread calling fetch_summary_context will retry.
                pass

    @property
    def global_summary(self) -> str:
        """Read-only property to get the current summary without triggering updates."""
        with self._lock:
            return self._cached_summary