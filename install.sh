pip install -r requirements.txt

#for videoxl


pip install -e "videoxl/.[train]"
pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
cd videoxl
pip install -r requirements.txt