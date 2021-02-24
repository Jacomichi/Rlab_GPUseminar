#include <iomanip>
#include <limits>
#include <algorithm>
#include <cassert>

#include "thrust_all.cuh"
#include "./utility/timer.cuh"
#include "./utility/random.cuh"

#include "./src/particle.cuh"
#include "./src/mesh.cuh"
#include "./src/list.cuh"
#include "./src/file.cuh"
#include "./src/md_host.cu"

int main(){
    //Simulation parameter
    int N = 1<<12;
    double rho = 0.8;
    double temperature = 1.0;
    double friction = 1.0;
    double large_sigma = 1.4;

    Atoms atoms(N,rho,temperature,friction,large_sigma);

    //simulation setting
    double dt = 0.002;
    int tau = 1.0/dt;
    int max_steps = 30000*tau;
    int stabilization_steps = 100*tau;
    int equilibration_steps = 1000*tau;
    Setting setting(N,rho,dt);
    std::cout << "L : " << setting.L << '\n';

    auto gen = createRandGenerator();
    atoms.set_diameter();
    atoms.random_uniform(gen,setting.L);
    //atoms.create_square_lattice(setting.L);
    atoms.fillVelo(0.0);


    std::cout << "finish setup" << '\n';
    //Output file setting
    int sampling_span = 100*tau;
    std::string dir_path_conf = create_outfile_conf_binary(atoms);
    //std::string dir_path_energy = create_outfile_energy_binary(atoms);
    Output outfile(dir_path_conf);
    //Output energy(dir_path_energy);

    //Timer setting
    cudaTimer timer;


    {//mesh setting
    double mesh_size = 2.3;
    int mesh_num = ceil(setting.L/mesh_size);
    assert(mesh_num > 3);
    Mesh mesh(N,mesh_num,setting.L);

    //List setting
    int list_size = 25;
    double list_cutoff = 2.0;
    unsigned int refresh_span = 100;//100回に1回更新
    List list(N,list_size,list_cutoff,refresh_span);
    std::cout << "start stabilization" << '\n';
    for(int t = 0; t < stabilization_steps; ++t){
        if( (t % list.refresh_span) == 0){
            h_update_verlet_list(atoms,list,mesh,setting);
        }
        // if( (t % (10*tau)) == 0){
        //     std::cout << "t : " << t/tau << '\n';
        //     //outfile.write_configuration(atoms);
        // }
        quench(atoms,list,mesh,setting);
    }
}


    double mesh_size = 3.5;
    int mesh_num = ceil(setting.L/mesh_size);
    assert(mesh_num > 3);
    Mesh mesh(N,mesh_num,setting.L);

    //List setting
    int list_size = 40;
    double list_cutoff = 3.5;
    unsigned int refresh_span = 100;//100回に1回更新
    List list(N,list_size,list_cutoff,refresh_span);



    std::cout << "start equlibration" << '\n';
    for(int t = 0; t < equilibration_steps; ++t){
        if( (t % list.refresh_span) == 0){
            h_update_verlet_list(atoms,list,mesh,setting);
        }
        EoM(atoms,list,mesh,setting,gen);
    }
    std::cout << "finish equilibration" << '\n';

    std::cout << "simulation start" << '\n';
    double ave_kin = 0.0;
    double ave_pot = 0.0;
    timer.start_record();

    for(int t = 0; t < max_steps; ++t){
        if( (t % list.refresh_span) == 0){
            //h_create_list_full_search(atoms,list,setting);
            h_update_verlet_list(atoms,list,mesh,setting);
            // ave_kin = 0.0;
            // ave_kin = atoms.average_kinetic();
            // ave_pot = h_calc_potential_energy(atoms,setting,list);
            // std::cout << "step : " << t << " kinetic energy : " << ave_kin << " average potential : " << ave_pot << '\n';
            //std::cout << "step : " << t << " kinetic energy : " << ave_kin  << '\n';
            //outfile.write_configuration(atoms);
        }
        if( (t % sampling_span) == 0){
            // ave_kin = 0.0;
            // ave_kin = atoms.average_kinetic();
            // ave_pot = h_calc_potential_energy(atoms,setting,list);
            // std::cout << "step : " << t << " kinetic energy : " << ave_kin << " average potential : " << ave_pot << '\n';
            //std::cout << "step : " << t << " kinetic energy : " << ave_kin  << '\n';
            outfile.write_configuration(atoms);
        }
        EoM(atoms,list,mesh,setting,gen);
    }
    std::cout << "finish simulation" << '\n';


    timer.stop_record();
    timer.print_result();

    curandDestroyGenerator(gen);

    return 0;
}
