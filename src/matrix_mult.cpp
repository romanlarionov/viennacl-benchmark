
/* Written by: Roman Larionov */

#include <iostream>
#include <vector>

#include "viennacl/matrix.hpp"
#include <viennacl/tools/timer.hpp>
#include "viennacl/tools/random.hpp"

void perform_matrix_multiplication(int rows, int cols, std::vector<std::vector<double> > &result, double &total_gflops)
{
    viennacl::matrix<double>::size_type dim_X = rows;
    viennacl::matrix<double>::size_type dim_Y = cols;

    std::vector<std::vector<double> > host_matrix1;
    std::vector<std::vector<double> > host_matrix2;
    viennacl::matrix<double> device_matrix1(dim_X, dim_Y);
    viennacl::matrix<double> device_matrix2(dim_X, dim_Y);

    viennacl::tools::uniform_random_numbers<double> random_number_generator;

    // fill matrices
    for (std::size_t i = 0; i < dim_X; ++i)
    {
        std::vector<double> curr_vec1;
        std::vector<double> curr_vec2;
        host_matrix1.push_back(curr_vec1);
        host_matrix2.push_back(curr_vec2);

        for (std::size_t j = 0; j < dim_Y; ++j)
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
    viennacl::backend::finish();

    // https://github.com/viennacl/viennaclbench-dev/blob/master/src/benchmarks/benchmark_blas3.cpp#L118
    total_gflops = (device_matrix1.size1() / 1000.0) *
                   (device_matrix1.size2() / 1000.0) *
                   (device_matrix2.size2() / 1000.0) * 2.0;

    // transfer result back
    viennacl::copy(device_result, host_matrix1);
    result = host_matrix1;
}

int main(int argc, char *argv[])
{
    int i = 0;
    int num_runs = 10;
    int rows = 4000;
    int cols = 4000;

    if (argc > 1) {
        num_runs = std::atoi(argv[1]);
    	if (argc == 4) {
        	rows = std::atoi(argv[2]);
        	cols = std::atoi(argv[3]);
    	}
	}

    std::ofstream output;
    output.open("matrix_mult.txt");

    viennacl::tools::timer timer;
    double exec_time = 0.0;
    double total_exec_time = 0.0;
    double total_gflops = 0.0;

    // loop for total number of runs or until 20 minutes has passed
    for (i = 1; (i <= num_runs) || (total_exec_time >= 1200.0); ++i)
    {
        std::vector<std::vector<double> > result;
        double curr_gflops = 0.0;

        timer.start();
        perform_matrix_multiplication(rows, cols,  result, curr_gflops);
        exec_time = timer.get();
        total_exec_time += exec_time;
        total_gflops += curr_gflops;

		if (i % 10 == 0) std::cout << i << std::endl;

        output << "Run : " << i << "\t\t";
        output << "Real execution time: " << exec_time << "s\t";
        output << "GFLOPs per second: " << (curr_gflops / exec_time) << "\n";
        // not displaying result as it is very large
    }

    output << "\nMean Execution Time: " << (total_exec_time / (i - 1)) << "s\n";
    output << "Mean GFLOPs per second: : " << (total_gflops / total_exec_time) << "\n";
    output << "Total Execution Time: " << total_exec_time << "s\n";
    output.close();
}


