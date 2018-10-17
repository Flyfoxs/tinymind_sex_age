cd "$(dirname "$0")"

cd ..

rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/tinymind_3/output/best/*all*.h5 ./output/best/
rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/tinymind_3/sub/best/*.csv ./sub/best/
rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/tinymind_3/output/best/*.h5 ./output/best/
rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/tinymind_3/cache/*1011*.csv ./cache/
#rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/tinymind_3/cache/*.pkl ./cache/

