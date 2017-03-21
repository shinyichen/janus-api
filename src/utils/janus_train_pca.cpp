#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>

#include "iarpa_janus.h"
#include "iarpa_janus_io.h"



using namespace std;

const char *get_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}


void printUsage()
{
    printf("Usage: janus_train_pca sdk_path temp_path template_list_file output_pca_file [-algorithm <algorithm>] [-verbose]\n");
}

int main(int argc, char *argv[])
{
    int requiredArgs = 5;

    if ((argc < requiredArgs) || (argc > 7)) {
        printUsage();
        return 1;
    }

    char *algorithm = "";
    bool verbose = false;

    for (int i = 0; i < argc - requiredArgs; i++) {
        if (strcmp(argv[requiredArgs+i],"-algorithm") == 0)
            algorithm = argv[requiredArgs+(++i)];
        else if (strcmp(argv[requiredArgs+i],"-verbose") == 0)
            verbose = true;
        else {
            fprintf(stderr, "Unrecognized flag: %s\n", argv[requiredArgs+i]);
            return 1;
        }
    }

    //JANUS_CHECK(janus_train_pca_helper(argv[3], argv[4]))

    return EXIT_SUCCESS;
}
