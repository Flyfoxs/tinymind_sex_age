cd "$(dirname "$0")"

cd ..

rsync -av --exclude-from './bin/exclude.txt' ./ hdpsbp@ai-prd-07:/users/hdpsbp/felix/tinymind_3

#rsync -av ./cache/extend_time_span*.csv hdpsbp@ai-prd-07:/users/hdpsbp/felix/tinymind_3/cache/
