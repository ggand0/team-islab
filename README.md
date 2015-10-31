# team-islab
  
Installation:

- Download the main 'imgs' dir and put it to the project dir  
(https://www.kaggle.com/c/noaa-right-whale-recognition/data)
- Add w_7489.jpg to 'imgs'
- Make sure there's no '.DS_Store' files
- Install PIL lib:  
```bash
sudo pip install PIL --allow-external PIL --allow-unverified PIL
```
- Create sub dirs in 'imgs' based on whale labels:
```bash
python create_dirs.py
```
- Generate sample annotated json data
```bash
python image_parser_mod.py
```
- See sample cropped data
```bash
python crop_from_json.py
```
