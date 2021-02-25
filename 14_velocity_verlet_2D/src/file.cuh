#ifndef FILE_CUH
#define FILE_CUH

#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <sstream>
#include "./particle.cuh"
#include "../thrust_all.cuh"




std::string create_data_directory(std::string dir_name){
	char tmp[256];
  getcwd(tmp, 256);
	std::string cwd = tmp;
	std::cout << "Current working directory: " << cwd << "\n";
  auto path_dir = cwd + "/" + dir_name;
	struct stat statBuf;

  if(stat(path_dir.c_str(),&statBuf)){
		mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
		int k = mkdir(dir_name.c_str(), mode);
		if (k != 0){
				printf("We could not make directory\n");
			}
  }
  return path_dir;
}

int existFile(std::string path,std::string filename)
{
    struct stat st;
		std::string file_path = path + "/" + filename;
		const char* cpath = file_path.c_str();

    if (stat(cpath, &st) != 0) {
        return 0;
    }

    // ファイルかどうか
    // S_ISREG(st.st_mode); の方がシンプルだが、Visual Studio では使えない。
    return (st.st_mode & S_IFMT) == S_IFREG;
}

std::string create_outputfilename_binary(Atoms atoms,std::string filename_head,std::string path_dir){
  std::string N = std::to_string(atoms.N);
  std::string rho = std::to_string(atoms.rho);
  std::string temperature = std::to_string(atoms.temperature);
  std::string friction = std::to_string(atoms.friction);
  std::string sigma_large = std::to_string(atoms.large_sigma);

	std::ostringstream ss;
	int id = 1;
	ss << std::setw(4) << std::setfill('0') << id;
	std::string data_index = ss.str();
	std::string filename = filename_head + "_N" + N + "_rho" + rho +"_T" + temperature + "_zeta" + friction + "_sigmaL" + sigma_large + "_" + data_index +  ".dat";

	while(existFile(path_dir,filename)){
		id += 1;
		ss.str("");
	  ss << std::setw(4) << std::setfill('0') << id;
		data_index = ss.str();
	  filename = filename_head + "_N" + N + "_rho" + rho +"_T" + temperature + "_zeta" + friction + "_sigmaL" + sigma_large + "_" + data_index +  ".dat";
	}

  return filename;
}

std::string create_outfile_conf_binary(Atoms atoms){
  auto path_dir = create_data_directory("configuration");
  auto filename = create_outputfilename_binary(atoms,"conf",path_dir);
  return path_dir + "/" + filename;
}

std::string create_outfile_energy_binary(Atoms atoms){
  auto path_dir = create_data_directory("energy");
  auto filename = create_outputfilename_binary(atoms,"energy",path_dir);
  return path_dir + "/" + filename;
}


struct Output{
  FILE *out_file;

  Output(std::string filename){
    out_file = fopen(filename.c_str(),"w");
  }

  ~Output(){
    fclose(out_file);
  }

	void write_configuration(const Atoms &atoms){
			for(int i =0; i< atoms.N; ++i){
					double x = atoms.x[i];
					double y = atoms.y[i];
					fprintf(out_file,"%e %e\n",x,y);
			}
	}

  void write_average_energy(Atoms atoms,double time){
    double ave_kin = atoms.average_kinetic();
    double ave_pot = atoms.average_potential();
    fprintf(out_file, "%15.7e %15.7e %15.7e\n",time, ave_kin,ave_pot);
  }


};


#endif
