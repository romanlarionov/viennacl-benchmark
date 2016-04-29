
/* Written by: Roman Larionov */

#include <iostream>
#include <vector>

#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/tools/random.hpp"
#include "viennacl/tools/timer.hpp"

void perform_inner_product(int num_elements, double &result)
{
    std::size_t length = static_cast<std::size_t>(num_elements);

    std::vector<double> host_vec1;
    std::vector<double> host_vec2;
    viennacl::vector<double> device_vec1(length);
    viennacl::vector<double> device_vec2(length);

    viennacl::tools::uniform_random_numbers<double> random_number_generator;

    // fill vectors
    for (std::size_t i = 0; i < length; ++i)
    {
        host_vec1.push_back(random_number_generator() * 10);
        host_vec2.push_back(random_number_generator() * 10);
    }

    // transfer data to gpu
    viennacl::fast_copy(host_vec1, device_vec1);
    viennacl::fast_copy(host_vec2, device_vec2); // comment explaining why I chose things like this (fast_copy)

    // perform computation
    result = viennacl::linalg::inner_prod(device_vec1, device_vec2);
    viennacl::backend::finish();
}

int main(int argc, char *argv[])
{
    int i = 0;
    int num_runs = 10;
    int dimensionality = 3000000;

    if (argc > 1) {
        num_runs = std::atoi(argv[1]);
        if (argc == 3)
            dimensionality = std::atoi(argv[2]);
    }

    std::ofstream output;
    output.open("inner_product.txt");

    viennacl::tools::timer timer;
    double exec_time = 0;
    double total_exec_time = 0.0;

    // loop for total number of runs or until 20 minutes has passed
    for (i = 1; (i <= num_runs) || (total_exec_time >= 1200.0); ++i)
    {
        double result = 0.0;

        timer.start();
        perform_inner_product(dimensionality,  result);
        exec_time = timer.get();
        total_exec_time += exec_time;

        output << "Run : " << i << "\t";
        output << "Real execution time: " << exec_time << "s\t";
        output << "Result: " << result << "\n";
    }

    output << "\nMean Execution Time: " << (total_exec_time / (i - 1)) << "s\n";
    output << "Total Execution Time: " << total_exec_time << "s\n";
    output.close();
}
