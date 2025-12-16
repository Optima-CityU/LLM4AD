#include "Population.h"

namespace genvrp {

    void Population::generatePopulation() {
        for(int i = 0; i < 4 * params.ga.mu; i++) {
            Individual* randomIndiv = new Individual(&params);
            split.generalSplit(randomIndiv, params.data.nbVehicles);
            localSearch.run(randomIndiv, params.ga.penaltyCapacity, params.ga.penaltyDuration);
            addIndividual(randomIndiv, true);
            if(!randomIndiv->isFeasible && std::rand() % 2 == 0) // Repair half of the solutions in case of infeasibility
            {
                localSearch.run(randomIndiv, params.ga.penaltyCapacity * 10., params.ga.penaltyDuration * 10.);
                if(randomIndiv->isFeasible)
                    addIndividual(randomIndiv, false);
            }
            delete randomIndiv;
        }
    }

    bool Population::addIndividual(const Individual* indiv, bool updateFeasible) {
        if(updateFeasible) {
            listFeasibilityLoad.push_back(indiv->cost.capacityExcess < MY_EPSILON);
            listFeasibilityDuration.push_back(indiv->cost.durationExcess < MY_EPSILON);
            listFeasibilityLoad.pop_front();
            listFeasibilityDuration.pop_front();
        }

        // Find the adequate subpopulation in relation to the individual feasibility
        SubPopulation& subpop = (indiv->isFeasible) ? feasibleSubpopulation : infeasibleSubpopulation;

        // Create a copy of the individual and updade the proximity structures calculating inter-individual distances
        Individual* myIndividual = new Individual(*indiv);
        for(Individual* myIndividual2 : subpop) {
            double myDistance = myIndividual->brokenPairsDistance(myIndividual2);
            myIndividual2->indivsPerProximity.insert({myDistance, myIndividual});
            myIndividual->indivsPerProximity.insert({myDistance, myIndividual2});
        }

        // Identify the correct location in the population and insert the individual
        int place = (int)subpop.size();
        while(place > 0 && subpop[place - 1]->cost.penalizedCost > indiv->cost.penalizedCost - MY_EPSILON)
            place--;
        subpop.emplace(subpop.begin() + place, myIndividual);

        // Trigger a survivor selection if the maximimum population size is exceeded
        if((int)subpop.size() > params.ga.mu + params.ga.lambda)
            while((int)subpop.size() > params.ga.mu)
                removeWorstBiasedFitness(subpop);

        // Track best solution
        if(indiv->isFeasible && indiv->cost.penalizedCost < bestSolutionRestart.cost.penalizedCost - MY_EPSILON) {
            bestSolutionRestart = *indiv;
            if(indiv->cost.penalizedCost < bestSolutionOverall.cost.penalizedCost - MY_EPSILON) {
                bestSolutionOverall = *indiv;
            }
            return true;
        } else
            return false;
    }

    void Population::addIndividualLS(const Individual* indiv, bool updateFeasible) {
        // TV: I know, it's not the most efficient way of doing at all...
        // however, given how infrequently this function is used (once per decomposition) it's never going to be a bottlneck
        Individual* myIndividual = new Individual(*indiv);

        if(myIndividual->isFeasible) {
            localSearch.run(myIndividual, params.ga.penaltyCapacity, params.ga.penaltyDuration);
        } else {
            localSearch.run(myIndividual, params.ga.penaltyCapacity * 10., params.ga.penaltyDuration * 10.);
        }
        addIndividual(myIndividual, updateFeasible);
    }

    void Population::updateBiasedFitnesses(SubPopulation& pop) {
        // Ranking the individuals based on their diversity contribution (decreasing order of distance)
        std::vector<std::pair<double, int>> ranking;
        for(int i = 0; i < (int)pop.size(); i++)
            ranking.push_back({-pop[i]->averageBrokenPairsDistanceClosest(params.ga.nbCountDistMeasure), i});
        std::sort(ranking.begin(), ranking.end());

        // Updating the biased fitness values
        if(pop.size() == 1)
            pop[0]->biasedFitness = 0;
        else {
            for(int i = 0; i < (int)pop.size(); i++) {
                double divRank = (double)i / (double)(pop.size() - 1); // Ranking from 0 to 1
                double fitRank = (double)ranking[i].second / (double)(pop.size() - 1);
                if((int)pop.size() <= params.ga.nbElite) // Elite individuals cannot be smaller than population size
                    pop[ranking[i].second]->biasedFitness = fitRank;
                else
                    pop[ranking[i].second]->biasedFitness = fitRank + (1.0 - (double)params.ga.nbElite / (double)pop.size()) * divRank;
            }
        }
    }

    void Population::removeWorstBiasedFitness(SubPopulation& pop) {
        updateBiasedFitnesses(pop);
        if(pop.size() <= 1)
            throw std::string("Eliminating the best individual: this should not occur in HGS");

        Individual* worstIndividual = NULL;
        int worstIndividualPosition = -1;
        bool isWorstIndividualClone = false;
        double worstIndividualBiasedFitness = -1.e30;
        for(int i = 1; i < (int)pop.size(); i++) {
            bool isClone = (pop[i]->averageBrokenPairsDistanceClosest(1) < MY_EPSILON); // A distance equal to 0 indicates that a clone exists
            if((isClone && !isWorstIndividualClone) || (isClone == isWorstIndividualClone && pop[i]->biasedFitness > worstIndividualBiasedFitness)) {
                worstIndividualBiasedFitness = pop[i]->biasedFitness;
                isWorstIndividualClone = isClone;
                worstIndividualPosition = i;
                worstIndividual = pop[i];
            }
        }

        pop.erase(pop.begin() + worstIndividualPosition); // Removing the individual from the population
        for(Individual* myIndividual2 : pop)
            myIndividual2->removeProximity(worstIndividual); // Cleaning its distances from the other individuals in the population
        delete worstIndividual;                              // Freeing memory
    }

    void Population::restart() {
        std::cout << "----- RESET: CREATING A NEW POPULATION -----" << std::endl;
        for(Individual* indiv : feasibleSubpopulation)
            delete indiv;
        for(Individual* indiv : infeasibleSubpopulation)
            delete indiv;
        feasibleSubpopulation.clear();
        infeasibleSubpopulation.clear();
        bestSolutionRestart = Individual();
        generatePopulation();
    }

    void Population::managePenalties() {
        // Setting some bounds [0.1,1000] to the penalty values for safety
        double fractionFeasibleLoad = (double)std::count(listFeasibilityLoad.begin(), listFeasibilityLoad.end(), true) / (double)listFeasibilityLoad.size();
        if(fractionFeasibleLoad < params.ga.targetFeasible - 0.05 && params.ga.penaltyCapacity < 1000)
            params.ga.penaltyCapacity = std::min<double>(params.ga.penaltyCapacity * 1.2, 1000.);
        else if(fractionFeasibleLoad > params.ga.targetFeasible + 0.05 && params.ga.penaltyCapacity > 0.1)
            params.ga.penaltyCapacity = std::max<double>(params.ga.penaltyCapacity * 0.85, 0.1);

        // Setting some bounds [0.1,1000] to the penalty values for safety
        double fractionFeasibleDuration =
            (double)std::count(listFeasibilityDuration.begin(), listFeasibilityDuration.end(), true) / (double)listFeasibilityDuration.size();
        if(fractionFeasibleDuration < params.ga.targetFeasible - 0.05 && params.ga.penaltyDuration < 1000)
            params.ga.penaltyDuration = std::min<double>(params.ga.penaltyDuration * 1.2, 1000.);
        else if(fractionFeasibleDuration > params.ga.targetFeasible + 0.05 && params.ga.penaltyDuration > 0.1)
            params.ga.penaltyDuration = std::max<double>(params.ga.penaltyDuration * 0.85, 0.1);

        // Update the evaluations
        for(int i = 0; i < (int)infeasibleSubpopulation.size(); i++)
            infeasibleSubpopulation[i]->cost.penalizedCost = infeasibleSubpopulation[i]->cost.distance +
                                                             params.ga.penaltyCapacity * infeasibleSubpopulation[i]->cost.capacityExcess +
                                                             params.ga.penaltyDuration * infeasibleSubpopulation[i]->cost.durationExcess;

        // If needed, reorder the individuals in the infeasible subpopulation since the penalty values have changed (simple bubble sort for the sake of
        // simplicity)
        for(int i = 0; i < (int)infeasibleSubpopulation.size(); i++) {
            for(int j = 0; j < (int)infeasibleSubpopulation.size() - i - 1; j++) {
                if(infeasibleSubpopulation[j]->cost.penalizedCost > infeasibleSubpopulation[j + 1]->cost.penalizedCost + MY_EPSILON) {
                    Individual* indiv = infeasibleSubpopulation[j];
                    infeasibleSubpopulation[j] = infeasibleSubpopulation[j + 1];
                    infeasibleSubpopulation[j + 1] = indiv;
                }
            }
        }
    }

    Individual* Population::getBinaryTournament() {
        Individual* individual1;
        Individual* individual2;

        updateBiasedFitnesses(feasibleSubpopulation);
        updateBiasedFitnesses(infeasibleSubpopulation);

        int place1 = std::rand() % (feasibleSubpopulation.size() + infeasibleSubpopulation.size());
        if(place1 >= (int)feasibleSubpopulation.size())
            individual1 = infeasibleSubpopulation[place1 - feasibleSubpopulation.size()];
        else
            individual1 = feasibleSubpopulation[place1];

        int place2 = std::rand() % (feasibleSubpopulation.size() + infeasibleSubpopulation.size());
        if(place2 >= (int)feasibleSubpopulation.size())
            individual2 = infeasibleSubpopulation[place2 - feasibleSubpopulation.size()];
        else
            individual2 = feasibleSubpopulation[place2];

        if(individual1->biasedFitness < individual2->biasedFitness)
            return individual1;
        else
            return individual2;
    }

    Individual* Population::getBestFeasible() const {
        if(!feasibleSubpopulation.empty())
            return feasibleSubpopulation[0];
        else
            return NULL;
    }

    Individual* Population::getBestInfeasible() {
        if(!infeasibleSubpopulation.empty())
            return infeasibleSubpopulation[0];
        else
            return NULL;
    }

    Individual* Population::getBestFound() {
        if(bestSolutionOverall.cost.penalizedCost < 1.e29)
            return &bestSolutionOverall;
        else
            return NULL;
    }

    Individual Population::getEliteIndividual(double pct) const {
        assert(pct > 0.0);
        assert(pct <= 1.0);

        if(!feasibleSubpopulation.empty()) {
            const auto bound = static_cast<std::size_t>(std::max(feasibleSubpopulation.size() * pct, 1.0));
            const auto position = std::rand() % bound;
            auto indiv = *feasibleSubpopulation.at(position);
            return indiv;
        } else {
            // Make sure not both populations are empty.
            assert(!infeasibleSubpopulation.empty());
            const auto bound = static_cast<std::size_t>(std::max(infeasibleSubpopulation.size() * pct, 1.0));
            const auto position = std::rand() % bound;
            auto indiv = *infeasibleSubpopulation.at(position);
            return indiv;
        }
    }

    void Population::printState(int nbIter, int nbIterNoImprovement, const std::string& prefix) {
        std::printf("%s", prefix.c_str());
        std::printf("It %6d %6d | T(s) %.2f", nbIter, nbIterNoImprovement, (double)clock() / (double)CLOCKS_PER_SEC);

        if(getBestFeasible() != NULL)
            std::printf(" | Feas %zu %.2f %.2f", feasibleSubpopulation.size(), getBestFeasible()->cost.penalizedCost, getAverageCost(feasibleSubpopulation));
        else
            std::printf(" | NO-FEASIBLE");

        if(getBestInfeasible() != NULL)
            std::printf(" | Inf %zu %.2f %.2f", infeasibleSubpopulation.size(), getBestInfeasible()->cost.penalizedCost,
                        getAverageCost(infeasibleSubpopulation));
        else
            std::printf(" | NO-INFEASIBLE");

        std::printf(" | Div %.2f %.2f", getDiversity(feasibleSubpopulation), getDiversity(infeasibleSubpopulation));
        std::printf(" | Feas %.2f %.2f", (double)std::count(listFeasibilityLoad.begin(), listFeasibilityLoad.end(), true) / (double)listFeasibilityLoad.size(),
                    (double)std::count(listFeasibilityDuration.begin(), listFeasibilityDuration.end(), true) / (double)listFeasibilityDuration.size());
        std::printf(" | Pen %.2f %.2f", params.ga.penaltyCapacity, params.ga.penaltyDuration);
        std::cout << std::endl;
    }

    double Population::getDiversity(const SubPopulation& pop) {
        double average = 0.;
        int size = std::min<int>(params.ga.mu, pop.size()); // Only monitoring the "mu" better solutions to avoid too much noise in the measurements
        for(int i = 0; i < size; i++)
            average += pop[i]->averageBrokenPairsDistanceClosest(size);
        if(size > 0)
            return average / (double)size;
        else
            return -1.0;
    }

    double Population::getAverageCost(const SubPopulation& pop) {
        double average = 0.;
        int size = std::min<int>(params.ga.mu, pop.size()); // Only monitoring the "mu" better solutions to avoid too much noise in the measurements
        for(int i = 0; i < size; i++)
            average += pop[i]->cost.penalizedCost;
        if(size > 0)
            return average / (double)size;
        else
            return -1.0;
    }

    void Population::exportBKS(std::string fileName) {
        double readCost;
        std::vector<std::vector<int>> readSolution;
        std::cout << "----- CHECKING FOR POSSIBLE BKS UPDATE" << std::endl;
        bool readOK = Individual::readCVRPLibFormat(fileName, readSolution, readCost);
        if(bestSolutionOverall.cost.penalizedCost < 1.e29 && (!readOK || bestSolutionOverall.cost.penalizedCost < readCost - MY_EPSILON)) {
            std::cout << "----- NEW BKS: " << bestSolutionOverall.cost.penalizedCost << " !!!" << std::endl;
            bestSolutionOverall.exportCVRPLibFormat(fileName);
        }
    }

    Population::Population(Params& params, Split& split, LocalSearch& localSearch) : params(params), split(split), localSearch(localSearch) {
        listFeasibilityLoad = std::list<bool>(100, true);
        listFeasibilityDuration = std::list<bool>(100, true);
        generatePopulation();
    }

    Population::~Population() {
        for(int i = 0; i < (int)feasibleSubpopulation.size(); i++)
            delete feasibleSubpopulation[i];
        for(int i = 0; i < (int)infeasibleSubpopulation.size(); i++)
            delete infeasibleSubpopulation[i];
    }

} // namespace genvrp