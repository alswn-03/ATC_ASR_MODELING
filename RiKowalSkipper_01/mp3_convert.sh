#!/bin/bash

input_folder="/trimmed_audio_03" # .wav 파일들이 있는 폴더 경로
output_folder="/path/to/output/trimmed_audio_mp3_03" # 변환된 .mp3 파일을 저장할 폴더 경로

mkdir -p "$output_folder" # 출력 폴더가 없으면 생성

for file in "$input_folder"/*.wav; do
  filename=$(basename "$file" .wav) # 확장자 제거한 파일 이름
  ffmpeg -i "$file" -b:a 256k "$output_folder/$filename.mp3"
done

echo "변환 완료: $output_folder에 저장됨"
