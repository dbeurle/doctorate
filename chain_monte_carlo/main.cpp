
// To compile use
// clang++ -O3 -std=c++14 PolymerChainLength.cpp -o active_chains

#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

class polymer_network
{
public:
    polymer_network(std::size_t const number_of_active_chains,
                    std::size_t const lower_bound,
                    std::size_t const upper_bound)
        : generator(std::random_device{}()),
          real_dist{0.0, 1.0},
          active_population(number_of_active_chains),
          inactive_population(number_of_active_chains, 0)
    {
        std::uniform_int_distribution<std::int32_t> uniform_dist(lower_bound, upper_bound);

        std::generate(begin(active_population), end(active_population), [&]() {
            return uniform_dist(generator);
        });
    }

    /// Perform a scission event by producing a random event and choosing
    /// a chain at random based on the length of the polymer in comparison
    /// to the other active_chains in the system.  This means longer active_chains have a
    /// proportionally larger probability of undergoing a scission event
    void scission(double const scission_probability)
    {
        if (scission_probability > 1.0 || scission_probability < 0.0)
        {
            throw std::domain_error("The chain scission probability needs to be between 0 and 1, "
                                    "not "
                                    + std::to_string(scission_probability) + "\n");
        }

        // Check if an event will not occur, then return
        if (real_dist(generator) > scission_probability) return;

        // Generate a random integer between 0 and the total number of
        // segments in the network and find the parent chain.  This means larger
        // chains are more likely to experience a scission event.
        // These events can affect a chain that is no longer in use but this
        // will be slightly less likely since the already scissioned chain will
        // have one less bond to severe
        std::uniform_int_distribution<std::int64_t> segment_dist(std::int64_t{},
                                                                 active_population.size() - 1);

        auto const segment_to_scission = segment_dist(generator);

        auto const active_chains = active_population.at(segment_to_scission);
        auto const inactive_chains = inactive_population.at(segment_to_scission);

        std::uniform_int_distribution<std::int64_t> pop_dist(std::int64_t{},
                                                             active_chains + inactive_chains);

        if (active_chains > pop_dist(generator))
        {
            // scission the active set chain
            --active_population.at(segment_to_scission);
        }
        else
        {
            // scission the inactive set chain
            --inactive_population.at(segment_to_scission);
        }

        // Decide where the scission will occur in the chains
        auto const split = std::uniform_int_distribution<std::int64_t>{std::int64_t{},
                                                                       segment_to_scission}(generator);
        ++inactive_population.at(split);
        ++inactive_population.at(segment_to_scission - split);
    }

    void creation(double const creation_probability)
    {
        if (creation_probability > 1.0 || creation_probability < 0.0)
        {
            throw std::domain_error("The chain creation probability needs to be between 0 and 1, "
                                    "not "
                                    + std::to_string(creation_probability) + "\n");
        }

        // Check if an event will not occur, then return
        if (real_dist(generator) > creation_probability) return;

        std::cout << "!!!creation event!!!\n";

        return;

        // Here we crosslink the chains and perform the logic of how a chain
        // can form from each of the possibilities
        // 1) An active chain is joined to an inactive chain
        // 2) An active chain is joined to an active chain
        // 3) An inactive chain is joined to an inactive chain

        enum class events { active_active, active_inactive, inactive_inactive };
    }

    /// Print out the statistics of the network to file
    void statistics(std::int32_t const step)
    {
        std::cout << std::string(2, ' ') << "active chains: "
                  << std::accumulate(begin(active_population), end(active_population), std::int64_t{})
                  << "\n";
        std::cout << std::string(2, ' ') << "segments: " << active_population.size() << "\n";

        // Check the mass balance is maintained
        auto const segment_count = std::accumulate(begin(active_population),
                                                   end(active_population),
                                                   0l,
                                                   [index = 0l](auto sum, auto chain_number) mutable {
                                                       return sum + index++ * chain_number;
                                                   })
                                   + std::accumulate(begin(inactive_population),
                                                     end(inactive_population),
                                                     0l,
                                                     [index = 0l](auto sum, auto chain_number) mutable {
                                                         return sum + index++ * chain_number;
                                                     });

        std::cout << std::string(2, ' ') << "total segment count: " << segment_count << "\n";

        // // Print out the chain data to file
        // file.open("active_chains_" + std::to_string(step) + ".txt", std::fstream::out);
        //
        // for (std::size_t i{}; i < active_chains.size(); ++i)
        // {
        //     if (is_active[i])
        //     {
        //         file << active_chains[i] << std::endl;
        //     }
        // }
        // file.close();
    }

protected:
    std::mt19937 generator;
    std::uniform_real_distribution<> real_dist;
    std::fstream file;

    std::vector<std::int32_t> active_population;   /// Active number of chains with N segments
    std::vector<std::int32_t> inactive_population; /// Inactive number of chains with N segments
};

int main()
{
    auto constexpr number_of_active_chains = 100;
    auto constexpr monte_carlo_steps = 5'000;

    // Chain segment number lower and upper bound
    auto constexpr lower_bound = 1'000;
    auto constexpr upper_bound = 5'000;

    auto constexpr scission_rate = 0.9;
    auto constexpr creation_rate = 0.2;

    polymer_network network(number_of_active_chains, lower_bound, upper_bound);

    for (int step = 0; step < monte_carlo_steps; step++)
    {
        std::cout << "step " << step << "\n";
        network.scission(scission_rate);
        network.creation(creation_rate);

        network.statistics(step);
    }
    return 0;
}
