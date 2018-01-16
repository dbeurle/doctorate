
// To compile use
// clang++ -O3 -std=c++14 PolymerChainLength.cpp -o chains

#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <range/v3/action.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/core.hpp>
#include <range/v3/view.hpp>

class PolymerNetwork
{
public:
    PolymerNetwork(std::size_t const number_of_chains,
                   std::size_t const segments_mean,
                   std::size_t const segments_variance)
        : is_active(number_of_chains, true), chains(number_of_chains, 1)
    {
        this->generate_segments(segments_mean, segments_variance);
    }

    /**
     * Perform a scission event by producing a random event and choosing
     * a chain at random based on the length of the polymer in comparison
     * to the other chains in the system.  This means longer chains have a
     * proportionally larger probability of undergoing a scission event
     */
    void scission(double const scission_probability)
    {
        if (scission_probability > 1.0 || scission_probability < 0.0)
        {
            std::cout << "The chain scission probability needs to be between \
                          0 and 1, not "
                      << scission_probability << "\n";
            std::abort();
        }

        std::mt19937 generator(rd());

        std::uniform_real_distribution<> real_dist(0.0, 1.0);

        // Check if an event will not occur, then return
        if (real_dist(generator) > scission_probability) return;

        // Generate a random integer between 0 and the total number of
        // segments in the network and find the parent chain.  This means larger
        // chains are more likely to experience a scission event.
        // These events can affect a chain that is no longer in use but this
        // will be slightly less likely since the already scissioned chain will
        // have one less bond to severe
        std::uniform_int_distribution<> int_dist(0, ranges::accumulate(chains, 0));

        auto const segment_to_scission = int_dist(generator);

        // Find the random number in the length array. If the number is less
        // the number of segments then we found the owner chain
        auto accumulated_length = 0;
        auto chain = ranges::find_if(chains, [&](auto const& val) {
            accumulated_length += val;
            return segment_to_scission < accumulated_length;
        });

        if (chain != chains.end())
        {
            auto const i = std::distance(chains.begin(), chain);
            if (is_active[i])
            {
                is_active[i] = false;
                std::cout << "Changing active status of chain number " << i << "\n";
            }
            chains[i] = chains[i] > 0 ? chains[i] - 1 : 0;
        }
    }

    void compute_force()
    {
        // If the chain is broken then it will not contribute to the overall
        // force in the network.  Only accumulate the active chains in the network
    }

    /**
     * Print out the statistics of the network including:
     * 1) Number of active chains
     * 2)
     */
    void statistics(int step) const
    {
        using namespace ranges;

        auto const active_chains = accumulate(is_active, 0, [](auto const& sum, auto const& status) {
            return sum + (status ? 1 : 0);
        });

        std::cout << "Number of active chains " << active_chains << "\n";

        // Print out the chain data to file
        std::fstream file;
        file.open("chains_" + std::to_string(step) + ".txt", std::fstream::out);

        for_each(view::zip(chains, is_active), [&](auto const& tuple) {
            if (std::get<1>(tuple))
            {
                file << std::get<0>(tuple) << std::endl;
            }
        });
        file.close();
    }

protected:
    /**
     * Generate a uniform sampling of polymer chains between the specified
     * upper and lower bounds
     */
    void generate_segments(std::size_t const segments_mean, std::size_t const segments_variance)
    {
        using namespace ranges;
        // Generate the random chains of the chains in the system
        std::mt19937 generator(rd());
        std::normal_distribution<> normal_dist(segments_mean, segments_variance);

        chains |=
            action::transform([&](auto const& i) { return std::round(normal_dist(generator)); });
    }

protected:
    std::random_device rd;

    std::vector<bool> is_active; //!< Flag for active chain
    std::vector<int> chains;     //!< Length of polymer chain segments
};

int main()
{
    auto constexpr number_of_chains = 3'000;
    auto constexpr monte_carlo_steps = 13'000;

    auto constexpr segments_mean = 5'000;
    auto constexpr segments_variance = 1'000;

    PolymerNetwork network(number_of_chains, segments_mean, segments_variance);

    for (int step = 0; step < monte_carlo_steps; step++)
    {
        network.scission(0.9999999999999);

        // network.compute_force();

        network.statistics(step);
    }
    return 0;
}
