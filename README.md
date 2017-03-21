# janus-api

API for Janus deliveries. Currently using IARPA Phase II API.

# How To Build

cp /Users/jenniferchen/Documents/workspace_cpp/3rdParty ../
mkdir ./build
cd ./build
CMAKE ../
make

# How to Run

First create templates:

cd ./build
./src/utils/janus_create_templates <SDK-Dir> <TMP-Dir> <Data-Dir> <Template-CSV-Input> <Template-Output-Dir> <Template-Output-List> 0

SDK-Dir: /nfs/isicvlnas01/users/srawls/janus-isi-sdk/
Data-Dir: /lfs2/glaive/data/CS3_2.0/


Then create a gallery:
./src/utils/janus_create_gallery <SDK-Dir> <TMP-Dir> <Template-CSV-File> <Output-Gal-File> -algorithm noinit

Then search the gallery:
./src/utils/janus_search <SDK-Dir> <TMP-Dir> <Probe-template-CSV-File> <Gal-template-CSV-File> <Gal-File> <Max-Search-Results> <Output-Candidate-List-File> -algorithm noinit
