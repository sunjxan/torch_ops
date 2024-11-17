rm -rf build
python3 setup.py build
rm op/*.so
mv build/lib.*/*.so op
python3 test.py
