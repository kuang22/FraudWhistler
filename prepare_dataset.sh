wget -c -O VCTK.tar.gz https://datashare.ed.ac.uk/bitstream/handle/10283/2119/VCTK-Corpus.tar.gz?sequence=1&isAllowed=y
tar -zxf VCTK.tar.gz
rm -rf dataset
mkdir dataset
dir=$(pwd)
ln -sf "$dir/VCTK-Corpus/wav48" "$dir/dataset/ori"
wget -c https://www.openslr.org/resources/28/rirs_noises.zip
unzip -q rirs_noises.zip
sed -i "s@pwd_loc@${dir}@g" reverb.csv
