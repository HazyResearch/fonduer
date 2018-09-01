echo "Downloading fonduer test data..."
#url=http://i.stanford.edu/hazy/share/fonduer/fonduer_test_data_v0.3.0.tar.gz
data_tar=fonduer_test_data_v0.3.0
if type wget &>/dev/null; then
    wget -v -O fonduer_test_data_v0.3.0.tar.gz -L https://stanford.box.com/shared/static/o9zdfdajkqx5cqwxt8cf8s1hwtcncmlm.gz
fi

echo "Unpacking fonduer test data..."
tar -zxvf $data_tar.tar.gz -C data

echo "Deleting tar file..."
rm $data_tar.tar.gz

echo "Done!"
