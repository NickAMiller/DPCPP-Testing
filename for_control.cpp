#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>

class Stencil_kernel;
using namespace cl;

int main(void) {

    //Device selection
    //We will explicitly compile for the FPGA_EMULATOR, CPU_HOST, or FPGA
    #if defined(FPGA_EMULATOR)
        sycl::intel::fpga_emulator_selector device_selector;
    #elif defined(CPU_HOST)
        sycl::host_selector device_selector;
    #else
        sycl::intel::fpga_selector device_selector;
    #endif

    //Create queue
    auto property_list = sycl::property_list{ sycl::property::queue::enable_profiling() };
    sycl::queue device_queue(device_selector, NULL, property_list);
    sycl::event queue_event;

    //create a buffer that goes to the fpga
    int* arrayForBuffer = new int[1];
    int* controlvar = new int[1];

    arrayForBuffer[0] = 0;
    controlvar[0] = 5;

    //Buffer setup
    //Define the sizes of the buffers
    //The sycl buffer creation expects a type of sycl:: range for the size
    sycl::range<1> num_array{ 1 };

    sycl::buffer<int, 1> input_buffer(arrayForBuffer, num_array);
    sycl::buffer<int, 1> var_buffer(controlvar, num_array);


    //Device queue submit
    queue_event = device_queue.submit([&](sycl::handler& cgh) {

        //sycl::stream os(1024, 1024, cgh);

        //Create accessors
        auto accessor_buffer = input_buffer.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_var = var_buffer.get_access<sycl::access::mode::read_write>(cgh);

        cgh.single_task<class Stencil_kernel>([=]() {

            for (int i = 0; i < accessor_var[0]; i++)
                accessor_buffer[0] += 1;

        });

    });


    printf("Control Var: %d ; Buffer: %d\n", controlvar[0], arrayForBuffer[0]);

}

// Control Var: 5 ; Buffer: 0