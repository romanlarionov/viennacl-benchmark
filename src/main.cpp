
/* Written by: Roman Larionov */

#include <iostream>
#include <vector>

#include "hayai.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/tools/random.hpp"

void perform_inner_product(std::size_t num_elements, double &result)
{
    std::vector<double> host_vec1;
    std::vector<double> host_vec2;
    viennacl::vector<double> device_vec1(num_elements);
    viennacl::vector<double> device_vec2(num_elements);

    viennacl::tools::uniform_random_numbers<double> random_number_generator;

    // fill vectors
    for (std::size_t i = 0; i < num_elements; ++i)
    {
        host_vec1.push_back(random_number_generator() * 10);
        host_vec2.push_back(random_number_generator() * 10);
    }

    // transfer data to gpu
    viennacl::fast_copy(host_vec1, device_vec1);
    viennacl::fast_copy(host_vec2, device_vec2); // comment explaining why I chose things like this (fast_copy)

    // perform computation
    result = viennacl::linalg::inner_prod(device_vec1, device_vec2);
}

void perform_matrix_multiplication(std::size_t rows, std::size_t cols, std::vector<std::vector<double> > &result)
{
    viennacl::matrix<double>::size_type dim_X = rows;
    viennacl::matrix<double>::size_type dim_Y = cols;

    std::vector<std::vector<double> > host_matrix1;
    std::vector<std::vector<double> > host_matrix2;
    viennacl::matrix<double> device_matrix1(dim_X, dim_Y);
    viennacl::matrix<double> device_matrix2(dim_X, dim_Y);

    viennacl::tools::uniform_random_numbers<double> random_number_generator;

    // fill matrices
    for (std::size_t i = 0; i < rows; ++i)
    {
        std::vector<double> curr_vec1;
        std::vector<double> curr_vec2;
        host_matrix1.push_back(curr_vec1);
        host_matrix2.push_back(curr_vec2);

        for (std::size_t j = 0; j < cols; ++j)
        {
            host_matrix1[i].push_back(random_number_generator() * 10);
            host_matrix2[i].push_back(random_number_generator() * 10);
        }
    }

    // transfer data to gpu
    viennacl::copy(host_matrix1, device_matrix1);
    viennacl::copy(host_matrix2, device_matrix2); // try fast_copy?

    // perform computation
    viennacl::matrix<double> device_result = viennacl::linalg::prod(device_matrix1, device_matrix2);

    // transfer result back
    viennacl::copy(device_result, host_matrix1);
    result = host_matrix1;
}

int main()
{
    // add command line args for bash script
    int num_runs = 1;
    std::size_t dimensionality = 15;

    for (int i = 0; i < num_runs; ++i)
    {
        // benchmarking code

        std::cout << "Run number: " << i << "\n";
        double inner_product_result = 0;
        perform_inner_product(dimensionality, inner_product_result);

        // change to output to file
        std::cout << inner_product_result << "\n";

        // benchmarking code

        std::vector<std::vector<double> > matrix_mult_result;
        perform_matrix_multiplication(dimensionality, dimensionality, matrix_mult_result); // add support for multi-dimensionality

        // change to output to file
        for (std::size_t i = 0; i < dimensionality; ++i)
        {
            for (std::size_t j = 0; j < dimensionality; ++j)
                std::cout << matrix_mult_result[i][j] << " ";
            std::cout << "\n";
        }
        std::cout << "done\n";

        // maybe perform additional tests
    }

    // output benchmark scores to file

    return 0;
}