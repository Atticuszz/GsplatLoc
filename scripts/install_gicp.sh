pip uninstall small_gicp
cd /home/atticuszz/DevSpace/python/AutoDrive_backend/tests/small_gicp
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
cd ..
pip install .
