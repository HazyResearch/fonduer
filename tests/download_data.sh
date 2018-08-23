echo "Downloading fonduer test data..."
url=http://i.stanford.edu/hazy/share/fonduer/fonduer_test_data_v0.3.0.tar.gz
data_tar=fonduer_test_data_v0.3.0
if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N $url
fi

mkdir -p data
echo "Unpacking fonduer test data..."
tar -zxvf $data_tar.tar.gz -C data

echo "Deleting tar file..."
rm $data_tar.tar.gz

echo "Done!"
