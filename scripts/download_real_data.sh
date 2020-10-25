#!/bin/bash
# Downloads all the data needed from OneDrive and organizes it. Note that this could take up to 3 hours to download and split the videos

data_path=../data/full

sudo apt install unzip
sudo apt install zip

cd $data_path
mkdir frames

# Download the videos. Takes 20 min
curl 'https://canadaeast1-mediap.svc.ms/transform/zip?cs=fFNQTw' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'Content-Type: application/x-www-form-urlencoded' -H 'Origin: https://ubcca-my.sharepoint.com' -H 'DNT: 1' -H 'Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' --data-raw 'zipFileName=raw_data.zip&guid=f7a3f50a-e891-4934-926d-6893c874eada&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%22raw_data%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4BTXS3QEWCA5BCLPKUWLTMRZ67N%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwMzYzNzk5OSIsImV4cCI6IjE2MDM2NTk1OTkiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJzaWduaW5fc3RhdGUiOiJbXCJrbXNpXCJdIiwibmFtZWlkIjoiMCMuZnxtZW1iZXJzaGlwfG1lbm9uaUBzdHVkZW50LnViYy5jYSIsIm5paSI6Im1pY3Jvc29mdC5zaGFyZXBvaW50IiwiaXN1c2VyIjoidHJ1ZSIsImNhY2hla2V5IjoiMGguZnxtZW1iZXJzaGlwfDEwMDMyMDAwZWQ0MGYzYjlAbGl2ZS5jb20iLCJ0dCI6IjAiLCJ1c2VQZXJzaXN0ZW50Q29va2llIjoiMyJ9.QnBpZkR2ejdKc0lYczZQalY3MGlweTRhb3BXb05wNUNvY2Q3bjVndnJFRT0%22%2C%22isFolder%22%3Atrue%7D%5D%7D&oAuthToken=' -o raw_data.zip

# Fix a corrupted file issue. Takes 10 min
zip -FFv raw_data.zip --out raw_data_fixed.zip

unzip raw_data_fixed.zip

# Download labels.json
curl 'https://ubcca-my.sharepoint.com/personal/menoni_student_ubc_ca/_layouts/15/download.aspx?UniqueId=3fd66112%2D8f1a%2D46e4%2D88b7%2Dbe446a7981f8' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'Referer: https://ubcca-my.sharepoint.com/personal/menoni_student_ubc_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmenoni%5Fstudent%5Fubc%5Fca%2FDocuments%2FProjectX' -H 'DNT: 1' -H 'Connection: keep-alive' -H 'Cookie: MicrosoftApplicationsTelemetryDeviceId=171e66aa-4224-40e9-837a-219614dea782; MicrosoftApplicationsTelemetryFirstLaunchTime=2020-10-25T17:05:53.391Z; rtFa=JltMi3xCGHNTZTgHdeI7+TWQemU9NSoK6orLvCRMb+UmMkZGRjA4QzktOTFENC00RkM4LUJCREQtREQ1OUI3NDE0RERCC+JSC4QfH+M8Uqf6SGdYmzFja2LjWOWDvOFFXzeBEwlwfs6oIrc8vcr0PT7dVeKGQ5M+KjFQytvAROGvGycGgFAmGqbejjyO0SF6s4xFSnyW1RjeNkOs9jkbHNfsSX3btQbhqTMZYSV0H+7msow3SDhhhrhFJSUmLYu9oyiOaZLMx3N3D+UjAjBdNhM3J4r4hP25J5X5Vy8zf806oDBMO8lKpnSw7o8SOqBiCK0+Bj4BzK46uANiqt4hS48KBX+Sst086pO0hjqrqq1EwRowBSLYzeC/zeInmIm4Dwqwwg8eUTiU4SFMXVPAmxL6JfM0LvXmOHxzwHrp52cF6n5CN0UAAAA=; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjgsMGguZnxtZW1iZXJzaGlwfDEwMDMyMDAwZWQ0MGYzYjlAbGl2ZS5jb20sMCMuZnxtZW1iZXJzaGlwfG1lbm9uaUBzdHVkZW50LnViYy5jYSwxMzI0ODExOTEwNjAwMDAwMDAsMTMyNDcxMDkwOTkwMDAwMDAwLDEzMjQ4NTUxMTQ5MDQwNjQyMCwxNzIuODMuNDAuMTAzLDMsMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiLCw1NzUzZDFkYy0yODU4LTRlOWEtYjhiNy03MmU5YWQyMDY0NTQsNjAyZTg3OWYtMjAxOC1iMDAwLTlkMzYtZTkyMGQ0NWJjMTI2LDYwMmU4NzlmLTIwMTgtYjAwMC05ZDM2LWU5MjBkNDViYzEyNiwsMCwxMzI0ODEyMjc0ODk5MjgzMTIsMTMyNDgzNzgzNDg5OTI4MzEyLCwsZXlKNGJYTmZZMk1pT2lKYlhDSkRVREZjSWwwaWZRPT0sMjY1MDQ2Nzc0Mzk5OTk5OTk5OSwxMzI0ODExOTE0ODAwMDAwMDAsMzIxY2NmNjItNDcyZS00N2FlLTkwOGUtMDIxNGEzNTMxOTA1LG1rUmF0cWFBUlB1ekdRc0pjY0xDZHp6K0xTRVRaWGVhK3FwUG5QQ0VnTXlKOW5ObVAzUHV4MXkvOEkwVURFZEEyVXR0T3NDWFhRNFQ5emtXbm5jL2lUR0xnOVNBVTlDNTgyZHlqOWxlWU9TdXQ1QkRZUUlKU2hlV3M4QUJBUVZUTCtRbHZvWk9CL0NuL1BSVkdQZzlVM1NwUThaL0tWVDdZWXBnTlB1V1BCb1J1ejdTNm4zaUNuMkVJbWkvK2pIWDJRNzI0blZWeDI1OUFiTjZtVDByTExCeVh5d0R6WHZIdFlacDMyRmNWTnNUOGdGUzI0dlVwVVJqS1NkY01CcjAwdTdnQVRFQm1HcGlqcDlNd2ZPSjR0cmd0MXQxRStRWFA2bjhTZ0wzV3p3MVg5dTBBTG9iNHBrMk90ZGNQc0lEQzhiNTM2cm5SVmk4cEVINy9LemJqdz09PC9TUD4=; CCSInfo=MTAvMjgvMjAyMCA3OjEwOjQ3IFBNotftyul2YuuD8jdjSI0b94Oki9PhuiWqNfQPlqQwdRl1pGbYLVcH/UX43wVFLSgi8flfuf3d8N2tAneG+Y983FVVPpnuenOMtvrd92SP2i5eOnnMLjuVpzq/E/anNIaA90PQC37b+LkXKYVZG+yDf7tQNH9xuXXbOXvnF+S6Radpjfs80Vsf6G2onYb35i4KU09nIym3Uq6e3Nqi8dZVw3c4B7ANJehtXcrwrTbYANrH3dLXIP+QMabJUjto8eBExHdbxI0p5KZJNEYryptjwcUsk2Z/kmyF0wijHgFwo+5r+r/Uc0FQUU0hZlLymnXDTDI9wKbqKNN+0bQmoMYwbhUAAAA=; odbn=1; cucg=0' -H 'Upgrade-Insecure-Requests: 1' -H 'TE: Trailers' -o labels.json

cd ../../scripts

# Splits the videos into frames. Takes 1-2h depending on how many frames you want per video
python video_splitter.py

cd ../src/tools

python make_real_data_json.py
