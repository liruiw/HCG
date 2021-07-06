# Download the example replay buffer
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1vdkgXMOjEo_CJDqAO3dlJAneb0KzbK26' -O data.zip
echo "Data downloaded. Starting to unzip"
unzip data.zip  -d data/offline_data/
rm data.zip
