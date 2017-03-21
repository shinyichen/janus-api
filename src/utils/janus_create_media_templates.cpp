#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cstdlib>

#include "iarpa_janus.h"
#include "iarpa_janus_io.h"

const char *get_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}

void printUsage()
{
    printf("Usage: janus_create_templates sdk_path temp_path data_path metadata_file min_face_size templates_dir template_list_file template_role [-algorithm <algorithm>] [-verbose]\n");
}

int main(int argc, char *argv[])
{
    int requiredArgs = 8;
    if ((argc < requiredArgs) || (argc > 12)) {
        printUsage();
        return 1;
    }

    const char *ext1 = get_ext(argv[4]);
    if (strcmp(ext1, "csv") != 0) {
        printf("metadata_file must be \".csv\" format.\n");
        return 1;
    } 

    char *algorithm = "";
    bool verbose = false;
    int gpu_dev = 0;

    for (int i = 0; i < argc - requiredArgs; i++) {
        if (strcmp(argv[requiredArgs+i],"-algorithm") == 0)
            algorithm = argv[requiredArgs+(++i)];
        else if (strcmp(argv[requiredArgs+i],"-verbose") == 0)
            verbose = true;
	else if (strcmp(argv[requiredArgs+i],"-gpu") == 0) {  
	  gpu_dev = std::atoi(argv[requiredArgs+i+1]);
	  std::cout << "Using gpu dev: " << gpu_dev << std::endl;
	  i++;
	}
        else {
            fprintf(stderr, "Unrecognized flag: %s\n", argv[requiredArgs+i]);
            return 1;
        }
    }

    JANUS_ASSERT(janus_initialize(argv[1], argv[2], algorithm, gpu_dev))
    JANUS_ASSERT(janus_media_templates_helper(argv[3], argv[4], atoi(argv[5]), argv[6], argv[7], static_cast<janus_template_role>(atoi(argv[8])), verbose))
    JANUS_ASSERT(janus_finalize())

    return EXIT_SUCCESS;
}
