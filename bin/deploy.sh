cd "$(dirname "$0")"

cd ..

rsync -av --exclude-from './bin/exclude.txt' ./ hdpsbp@ai-prd-07.cisco.com:/users/hdpsbp/felix/tinymind_3

