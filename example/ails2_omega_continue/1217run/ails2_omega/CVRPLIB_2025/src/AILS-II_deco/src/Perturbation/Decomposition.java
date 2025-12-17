/**
 * Decomposition作为Perturbation算子
 * 将问题分解为子问题，独立优化后合并
 */
package Perturbation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;

import Data.Instance;
import DiversityControl.OmegaAdjustment;
import Improvement.FeasibilityPhase;
import Improvement.IntraLocalSearch;
import Improvement.LocalSearch;
import SearchMethod.Config;
import SearchMethod.ConstructSolution;
import Solution.Node;
import Solution.Route;
import Solution.Solution;

/**
 * Decomposition扰动算子
 * 将当前问题分解为多个子问题，对每个子问题独立进行initial->local search（5000代），然后合并
 */
public class Decomposition extends Perturbation {
    
    private static final int LOCAL_SEARCH_ITERATIONS = 5000;
    
    
    
    /**
     * 分解策略接口
     */
    public interface DecompositionStrategy {
        /**
         * 将customers分解为多个子问题
         * @param customers 所有customer节点
         * @param numSubProblems 子问题数量
         * @return 每个子问题的customer集合列表
         */
        List<Set<Integer>> decompose(List<Node> customers, int numSubProblems);
    }
    
    /**
     * 随机分解策略（默认实现）
     */
    public static class RandomDecompositionStrategy implements DecompositionStrategy {
        private Random rand = new Random();
        
        @Override
        public List<Set<Integer>> decompose(List<Node> customers, int numSubProblems) {
            List<Set<Integer>> subProblems = new ArrayList<>();
            for (int i = 0; i < numSubProblems; i++) {
                subProblems.add(new HashSet<>());
            }
            
            // 随机将customers分配到子问题
            for (Node customer : customers) {
                int subProblemIdx = rand.nextInt(numSubProblems);
                subProblems.get(subProblemIdx).add(customer.name);
            }
            
            return subProblems;
        }
    }

    /**
     * 基于当前解的路由质心进行聚类分解（barycentre clustering）。
     * 思路：按非空路由的几何质心做 k-means，将同簇路由的客户合并为一个子问题。
     */
    public static class BarycentreClusteringStrategy implements DecompositionStrategy {
        private final Solution sol; // 当前总解（用于获取路由与客户归属）
        private final Instance instance; // 取坐标
        private final Random rnd = new Random();

        public BarycentreClusteringStrategy(Solution sol, Instance instance) {
            this.sol = sol;
            this.instance = instance;
        }

        private static final class Pt {
            final int routeIdx; final double x; final double y;
            Pt(int r, double x, double y) { this.routeIdx = r; this.x = x; this.y = y; }
        }

        @Override
        public List<Set<Integer>> decompose(List<Node> customers, int numSubProblems) {
            // 1) 汇集非空路由的质心
            List<Pt> pts = new ArrayList<>();
            for (int r = 0; r < sol.numRoutes; r++) {
                Route rt = sol.routes[r];
                if (rt == null || rt.numElements <= 1) continue; // 仅 depot 的路由跳过
                double sx = 0.0, sy = 0.0; int cnt = 0;
                Node it = rt.first.next;
                while (it != null && it != rt.first) {
                    if (it.name != rt.depot) {
                        Data.Point p = instance.getPoints()[it.name];
                        sx += p.x; sy += p.y; cnt++;
                    }
                    it = it.next;
                }
                if (cnt > 0) pts.add(new Pt(r, sx / cnt, sy / cnt));
            }

            if (pts.isEmpty()) {
                // 回退：无非空路由，则随机切分 customers
                return new RandomDecompositionStrategy().decompose(customers, numSubProblems);
            }

            int k = Math.min(numSubProblems, pts.size());
            if (k < 1) k = 1;

            // 2) 简单 k-means（欧氏距离，固定迭代次数）
            double[] cx = new double[k];
            double[] cy = new double[k];
            // 随机选择初始中心
            for (int i = 0; i < k; i++) {
                Pt p = pts.get(rnd.nextInt(pts.size()));
                cx[i] = p.x; cy[i] = p.y;
            }
            int n = pts.size();
            int[] assign = new int[n];
            for (int iter = 0; iter < 20; iter++) {
                // assign
                for (int i = 0; i < n; i++) {
                    double best = Double.POSITIVE_INFINITY; int bi = 0;
                    double px = pts.get(i).x, py = pts.get(i).y;
                    for (int c = 0; c < k; c++) {
                        double dx = px - cx[c], dy = py - cy[c];
                        double d2 = dx * dx + dy * dy;
                        if (d2 < best) { best = d2; bi = c; }
                    }
                    assign[i] = bi;
                }
                // update
                double[] sx = new double[k], sy = new double[k]; int[] cnt = new int[k];
                for (int i = 0; i < n; i++) { int c = assign[i]; sx[c] += pts.get(i).x; sy[c] += pts.get(i).y; cnt[c]++; }
                for (int c = 0; c < k; c++) if (cnt[c] > 0) { cx[c] = sx[c] / cnt[c]; cy[c] = sy[c] / cnt[c]; }
            }

            // 3) 将路由聚类结果映射为“客户集合”的子问题
            List<Set<Integer>> subProblems = new ArrayList<>();
            for (int c = 0; c < k; c++) subProblems.add(new HashSet<>());

            for (int i = 0; i < n; i++) {
                int cluster = assign[i];
                int r = pts.get(i).routeIdx;
                Route rt = sol.routes[r];
                Node it = rt.first.next;
                while (it != null && it != rt.first) {
                    if (it.name != rt.depot) subProblems.get(cluster).add(it.name);
                    it = it.next;
                }
            }

            // 移除空簇
            List<Set<Integer>> out = new ArrayList<>();
            for (Set<Integer> sp : subProblems) if (!sp.isEmpty()) out.add(sp);
            if (out.isEmpty()) {
                // 回退：避免空
                return new RandomDecompositionStrategy().decompose(customers, numSubProblems);
            }
            return out;
        }
    }
    
    private DecompositionStrategy decompositionStrategy;
    
    public Decomposition(Instance instance, Config config,
            HashMap<String, OmegaAdjustment> omegaSetup, IntraLocalSearch intraLocalSearch) {
        super(instance, config, omegaSetup, intraLocalSearch);
        this.perturbationType = PerturbationType.Decomposition;
        
        // 组件按子实例在局部创建
        
        // 使用默认的随机分解策略
        this.decompositionStrategy = new RandomDecompositionStrategy();
    }
    
    /**
     * 设置分解策略（允许未来扩展）
     */
    public void setDecompositionStrategy(DecompositionStrategy strategy) {
        this.decompositionStrategy = strategy;
    }
    
    @Override
    public void applyPerturbation(Solution s) {
        
        setSolution(s);
        
        // 1. 收集所有customers
        List<Node> customers = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            if (solution[i].nodeBelong) {
                customers.add(solution[i]);
            }
        }
        
        if (customers.isEmpty()) {
            assignSolution(s);
            return;
        }
        
        // 2. 确定子问题数量（与HGS-TV和ALNS一致：ceil(N / targetMaxSpCustomers)）
        // 如果targetMaxSpCustomers未设置或为0，使用默认值200（与HGS-TV和ALNS一致）
        int targetSz = (config != null) ? config.getTargetMaxSpCustomers() : 200;
        if (targetSz <= 0) {
            targetSz = 200;  // 默认值，与HGS-TV和ALNS一致
        }
        int numSubProblems = (int)Math.ceil(customers.size() / (double)targetSz);
        numSubProblems = Math.max(2, Math.min(numSubProblems, customers.size()));
        if (numSubProblems < 2) { assignSolution(s); return; }
        
        
        // 3. 分解问题（使用基于路由质心的分解策略）
        this.decompositionStrategy = new BarycentreClusteringStrategy(s, instance);
        List<Set<Integer>> subProblems = decompositionStrategy.decompose(customers, numSubProblems);
        
        // 过滤掉空的子问题
        List<Set<Integer>> validSubProblems = new ArrayList<>();
        for (Set<Integer> sp : subProblems) {
            if (!sp.isEmpty()) {
                validSubProblems.add(sp);
            }
        }
        
        if (validSubProblems.isEmpty()) {
            assignSolution(s);
            return;
        }
        
        // 4. 对每个子问题独立优化
        List<SubInstanceBundle> bundlesToMerge = new ArrayList<>();
        
        for (int spIdx = 0; spIdx < validSubProblems.size(); spIdx++) {
            Set<Integer> subProblemCustomers = validSubProblems.get(spIdx);
            
            // 简要日志：每个子问题优化开始
            
            // 为子问题创建独立的实例与解（方案A：真子实例 + 真子解）
            SubInstanceBundle subBundle = buildSubInstanceAndSolution(subProblemCustomers);
            Instance subInstance = subBundle.subInstance;
            Solution subSolution = subBundle.subSolution;
            
            // Initial: 构造初始解
            // 使用子实例对应的组件进行优化（包含子实例专属的 IntraLocalSearch）
            IntraLocalSearch spIntra = new IntraLocalSearch(subInstance, config);
            ConstructSolution spConstruct = new ConstructSolution(subInstance, config);
            FeasibilityPhase spFeasible = new FeasibilityPhase(subInstance, config, spIntra);
            LocalSearch spLocal = new LocalSearch(subInstance, config, spIntra);

            spConstruct.construct(subSolution);
            // 标记路由可浏览并补齐累积需求，确保 LS 有工作面
            for (int r = 0; r < subSolution.numRoutes; r++) {
                Route rt = subSolution.routes[r];
                if (rt != null && rt.numElements > 1) {
                    rt.setAccumulatedDemand();
                    rt.modified = true;
                }
            }
            // 先做一次可行化，再以其为基线
            spFeasible.makeFeasible(subSolution);
            for (int r = 0; r < subSolution.numRoutes; r++) {
                Route rt = subSolution.routes[r];
                if (rt != null && rt.numElements > 1) {
                    rt.setAccumulatedDemand();
                    rt.modified = true;
                }
            }
            // 计算 MP 上该簇的基线切片成本（以 MP 路由为口径）
            double mpSliceCost = computeMpSliceCost(s, instance, subProblemCustomers);
            
            // Local search: 持续5000代（含早停与守卫）
            int iteration = 0;
            double bestCost = subSolution.f;
            
            while (iteration < LOCAL_SEARCH_ITERATIONS) {
                // 若没有任何可浏览路由，提前退出，避免原地打转
                int modifiable = 0;
                for (int r = 0; r < subSolution.numRoutes; r++) {
                    Route rt = subSolution.routes[r];
                    if (rt != null && rt.numElements > 1 && rt.modified) modifiable++;
                }
                if (modifiable == 0) { break; }
                Solution prevSolution = new Solution(subInstance, config);
                prevSolution.clone(subSolution);
                double prevCost = prevSolution.f;
                
                // 仅在需要时再做可行化，避免把解不断拉远
                if (!subSolution.feasible()) {
                    spFeasible.makeFeasible(subSolution);
                    // 可行化后刷新路由标记与累积需求
                    for (int r = 0; r < subSolution.numRoutes; r++) {
                        Route rt = subSolution.routes[r];
                        if (rt != null && rt.numElements > 1) {
                            rt.setAccumulatedDemand();
                            rt.modified = true;
                        }
                    }
                }
                
                // Local search（基于子实例）
                spLocal.localSearch(subSolution, true);
                
                // 检查是否改进
                if (subSolution.f < prevCost - config.getEpsilon() &&
                    subSolution.f < bestCost - config.getEpsilon()) {
                    bestCost = subSolution.f;
                }
                // 早停：若当前轮无任何改进，则退出循环，避免原地重复
                if (Math.abs(subSolution.f - prevCost) <= config.getEpsilon()) { break; }
                
                iteration++;
            }
            
            // 以 MP 口径计算 SP 优化后切片成本（将子解映射回 MP 客户名并用 instance.dist 计）
            double spSliceCost = computeSpSliceCostMapped(subBundle, instance);
            double improvement = mpSliceCost - spSliceCost;
            
            // 仅当子问题切片在 MP 口径下严格改进时，才标记为待合并
            if (spSliceCost + config.getEpsilon() < mpSliceCost) {
                subBundle.subSolution = subSolution;
                bundlesToMerge.add(subBundle);
            }
        }
        
        // 5. 合并所有子问题的解到原Solution
        mergeSubSolutionsIntoOriginal(bundlesToMerge, s);
        
        // 6. 更新f和numRoutes
        f = s.f;
        numRoutes = s.numRoutes;
    }
    
    /**
     * 为子问题创建真正独立的Solution
     * - 只为属于子问题的customers创建独立的nodes
     * - 重建KNN（只包含子问题内的nodes）
     * - 确保完全独立，避免数据污染
     */
    private SubInstanceBundle buildSubInstanceAndSolution(Set<Integer> customerIds) {
        try {
            java.nio.file.Path tmp = java.nio.file.Files.createTempFile("subvrp-", ".vrp");
            tmp.toFile().deleteOnExit();

            // 构造子问题的顺序与映射：subIndex(1..m) -> mpName
            java.util.List<Integer> mpNames = new java.util.ArrayList<>();
            for (int name : customerIds) mpNames.add(name);

            // 写子问题VRP文件（EUC_2D）
            try (java.io.PrintWriter pw = new java.io.PrintWriter(java.nio.file.Files.newBufferedWriter(tmp))) {
                int m = mpNames.size();
                pw.println("NAME : SUBPROBLEM");
                pw.println("TYPE : CVRP");
                pw.println("DIMENSION : " + (m + 1));
                pw.println("EDGE_WEIGHT_TYPE : EUC_2D");
                pw.println("CAPACITY : " + instance.getCapacity());
                pw.println("NODE_COORD_SECTION");
                // index 1..m for customers; depot at index 0 in internal, but file needs 1-based ids; we output ids but parser ignores id's value
                // First write depot line
                Data.Point[] pts = instance.getPoints();
                Data.Point depotPt = pts[instance.getDepot()];
                pw.println("1 " + depotPt.x + " " + depotPt.y);
                // Then customers
                for (int i = 0; i < m; i++) {
                    Data.Point p = pts[mpNames.get(i)];
                    pw.println((i + 2) + " " + p.x + " " + p.y);
                }
                pw.println("DEMAND_SECTION");
                // depot demand 0
                pw.println("1 0");
                for (int i = 0; i < m; i++) {
                    Data.Point p = pts[mpNames.get(i)];
                    pw.println((i + 2) + " " + p.demand);
                }
                pw.println("DEPOT_SECTION");
                // depot index is 1-based in file; Instance will subtract 1
                pw.println("1");
                pw.println("EOF");
            }

            // 构建子实例
            SearchMethod.InputParameters ir = new SearchMethod.InputParameters();
            ir.readingInput(new String[]{"-file", tmp.toString(), "-rounded", "true"});
            Instance subInst = new Instance(ir);
            Solution subSol = new Solution(subInst, config);
            return new SubInstanceBundle(subInst, subSol, mpNames);
        } catch (Exception ex) {
            throw new RuntimeException("Failed to build sub instance", ex);
        }
    }
    
    // 旧的“在总实例上重建子KNN”的方法已不再需要（改为真子实例），因此移除。

    /** 子实例与解打包（含名称映射） */
    private static class SubInstanceBundle {
        final Instance subInstance;
        Solution subSolution;
        final java.util.List<Integer> subIndexToMpName;
        SubInstanceBundle(Instance subInstance, Solution subSolution, java.util.List<Integer> mapping) {
            this.subInstance = subInstance;
            this.subSolution = subSolution;
            this.subIndexToMpName = mapping;
        }
    }
    
    /**
     * 合并所有子问题的解到原始Solution中
     */
    private void mergeSubSolutionsIntoOriginal(List<SubInstanceBundle> subSolutions, Solution originalSolution) {
        // 1) 预取：记录每条原始路由上的客户顺序，以及待合并客户集合
        java.util.List<java.util.List<Integer>> originalRouteCustomers = new java.util.ArrayList<>();
        for (int r = 0; r < originalSolution.numRoutes; r++) {
            java.util.List<Integer> seq = new java.util.ArrayList<>();
            Route rt = originalSolution.routes[r];
            if (rt != null && rt.numElements > 1) {
                Node it = rt.first.next;
                while (it != null && it != rt.first) {
                    if (it.name != rt.depot) seq.add(it.name);
                    it = it.next;
                }
            }
            originalRouteCustomers.add(seq);
        }

        java.util.HashSet<Integer> mergedCustomers = new java.util.HashSet<>();
        for (SubInstanceBundle bundle : subSolutions) {
            for (int sp = 1; sp <= bundle.subIndexToMpName.size(); sp++) {
                mergedCustomers.add(bundle.subIndexToMpName.get(sp - 1));
            }
        }

        // 2) 清理原始解，使其成为空白接收器
        for (int i = 0; i < originalSolution.getNumRoutesMax(); i++) {
            originalSolution.routes[i].clean();
        }
        for (int i = 0; i < size; i++) {
            solution[i].prev = null;
            solution[i].next = null;
            solution[i].route = null;
            solution[i].nodeBelong = false;
            solution[i].modified = false;
        }

        int currentRouteIdx = 0;
        originalSolution.f = 0.0;

        // 3) 先放入改进过的子问题路由
        for (SubInstanceBundle bundle : subSolutions) {
            Solution subSol = bundle.subSolution;
            for (int i = 0; i < subSol.numRoutes; i++) {
                Route subRoute = subSol.routes[i];
                if (subRoute == null || subRoute.numElements <= 1) continue;
                if (currentRouteIdx >= originalSolution.getNumRoutesMax()) { break; }
                Route mergedRoute = originalSolution.routes[currentRouteIdx];
                mergedRoute.clean();
                Node it = subRoute.first.next;
                while (it != null && it != subRoute.first) {
                    if (it.name != subRoute.depot) {
                        int spName = it.name; // 1..m
                        int mpName = bundle.subIndexToMpName.get(spName - 1);
                        Node originalNode = originalSolution.getSolution()[mpName - 1];
                        if (originalNode != null) {
                            originalNode.prev = null;
                            originalNode.next = null;
                            originalNode.route = mergedRoute;
                            originalNode.modified = true;
                            originalNode.nodeBelong = true;
                            originalSolution.f += mergedRoute.addNodeEndRoute(originalNode);
                        }
                    }
                    it = it.next;
                }
                if (mergedRoute.numElements > 1) {
                    mergedRoute.setAccumulatedDemand();
                    mergedRoute.modified = true;
                    currentRouteIdx++;
                }
            }
        }

        // 4) 追加未参与改进的原始路由（按原顺序），确保所有客户均被覆盖
        for (java.util.List<Integer> seq : originalRouteCustomers) {
            // 过滤掉已合并客户后若仍有剩余，则按顺序构建一条路由
            java.util.List<Integer> remain = new java.util.ArrayList<>();
            for (int name : seq) if (!mergedCustomers.contains(name)) remain.add(name);
            if (remain.isEmpty()) continue;
            if (currentRouteIdx >= originalSolution.getNumRoutesMax()) break;
            Route r = originalSolution.routes[currentRouteIdx];
            r.clean();
            for (int name : remain) {
                Node originalNode = originalSolution.getSolution()[name - 1];
                if (originalNode == null) continue;
                originalNode.prev = null;
                originalNode.next = null;
                originalNode.route = r;
                originalNode.modified = true;
                originalNode.nodeBelong = true;
                originalSolution.f += r.addNodeEndRoute(originalNode);
            }
            if (r.numElements > 1) {
                r.setAccumulatedDemand();
                r.modified = true;
                currentRouteIdx++;
            }
        }

        // 5) 完成统计与清理
        originalSolution.numRoutes = currentRouteIdx;
        originalSolution.f = 0.0;
        for (int i = 0; i < originalSolution.numRoutes; i++) {
            if (originalSolution.routes[i] != null && originalSolution.routes[i].numElements > 1) {
                originalSolution.routes[i].fRoute = originalSolution.routes[i].F();
                originalSolution.f += originalSolution.routes[i].fRoute;
            }
        }
        originalSolution.removeEmptyRoutes();

        
    }
    
    /**
     * 设置Solution状态（重写以适配Decomposition的特殊需求）
     */
    @Override
    protected void setSolution(Solution s) {
        super.setSolution(s);
        
        // Decomposition使用omega参数来确定子问题数量
        if (omegaSetup != null && omegaSetup.containsKey(perturbationType + "")) {
            chosenOmega = omegaSetup.get(perturbationType + "");
            if (chosenOmega != null) {
                omega = chosenOmega.getActualOmega();
                omega = Math.min(omega, size);
            }
        } else {
            // 如果没有配置omega，使用默认值（子问题数量）
            omega = Math.max(2, Math.min(10, size / 50)); // 默认2-10个子问题
        }
    }

    // === 辅助：计算 MP 切片成本（基于 MP 当前解与 instance.dist） ===
    private double computeMpSliceCost(Solution mpSol, Instance inst, Set<Integer> clusterCustomers) {
        double cost = 0.0;
        for (int r = 0; r < mpSol.numRoutes; r++) {
            Route rt = mpSol.routes[r];
            if (rt == null || rt.numElements <= 1) continue;
            // 判断该路由是否完全属于 cluster（按客户集合）
            boolean allIn = true;
            Node it = rt.first.next;
            while (it != null && it != rt.first) {
                if (it.name != rt.depot && !clusterCustomers.contains(it.name)) { allIn = false; break; }
                it = it.next;
            }
            if (!allIn) continue;
            // 累加该路由在 MP 中的真实距离
            Node a = rt.first; Node b = rt.first.next;
            while (b != null && b != rt.first) { cost += inst.dist(a.name, b.name); a = b; b = b.next; }
            cost += inst.dist(a.name, rt.first.name);
        }
        return cost;
    }

    // === 辅助：计算 SP 优化解映射回 MP 客户名后的切片成本（用 instance.dist） ===
    private double computeSpSliceCostMapped(SubInstanceBundle bundle, Instance inst) {
        Solution spSol = bundle.subSolution;
        double cost = 0.0;
        for (int r = 0; r < spSol.numRoutes; r++) {
            Route rt = spSol.routes[r];
            if (rt == null || rt.numElements <= 1) continue;
            // 映射 route 上的连续边：spName -> mpName
            Node a = rt.first; Node b = rt.first.next;
            while (b != null && b != rt.first) {
                int mpA = (a.name == rt.depot) ? 0 : bundle.subIndexToMpName.get(a.name - 1);
                int mpB = (b.name == rt.depot) ? 0 : bundle.subIndexToMpName.get(b.name - 1);
                cost += inst.dist(mpA, mpB);
                a = b; b = b.next;
            }
            int mpLast = (a.name == rt.depot) ? 0 : bundle.subIndexToMpName.get(a.name - 1);
            int mpFirst = 0; // depot in MP
            cost += inst.dist(mpLast, mpFirst);
        }
        return cost;
    }
}

