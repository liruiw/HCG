# Download the pretrained model
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1oBfO4Xx0sSLyFMel6YDMQ5iQla6J8lIM' -O model_clutter.zip
echo "HCG Model downloaded. Starting to unzip"
mkdir output
unzip -q model_clutter.zip -d ./
rm model_clutter.zip


