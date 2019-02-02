echo "Downloading fonduer test data..."
url=http://i.stanford.edu/hazy/share/fonduer/fonduer_test_data_v0.3.0.tar.gz
data_tar=fonduer_test_data.tar.gz
if type curl &>/dev/null; then
    curl -RL --retry 3 -C - $url -o $data_tar
elif type wget &>/dev/null; then
    wget -N $url -O $data_tar
fi

echo "Unpacking fonduer test data..."
tar xvf $data_tar -C data

echo "Deleting tar file..."
rm $data_tar

echo "Done!"
