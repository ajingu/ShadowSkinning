# ShadowSkinning
Automatic shadow skinning

# Required Module
This project uses [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation).   
You should install **tf_pose** module according to [instructions](https://github.com/ildoonet/tf-pose-estimation#package-install).  

# Usage
### draw_contour.py
![contour](https://user-images.githubusercontent.com/20081122/46914566-9d14b400-cfda-11e8-8e2b-d20408556238.png)
  
### polygon_division.py
![polygon_division](https://user-images.githubusercontent.com/20081122/46914478-4ce92200-cfd9-11e8-8ab6-f1dfdf6c4d6d.png)
  
### draw_skeleton.py
![skeleton](https://user-images.githubusercontent.com/20081122/47789124-0f192700-dd57-11e8-8060-84647280fb8e.PNG)

### nearest_neighbour_skinning.py
![nearest](https://user-images.githubusercontent.com/20081122/47289631-94edf180-d636-11e8-90b2-aad39bc2e785.png)

![skinning](https://user-images.githubusercontent.com/20081122/47789131-117b8100-dd57-11e8-8157-54090a585416.PNG)


### draw_augmented_vertices_and_polygon.py
![augmented_polygon](https://user-images.githubusercontent.com/20081122/48066766-cc04fb00-e211-11e8-8334-dc9862b74478.PNG)

### shadowFactory.py
- Assign the directory path you want to watch to `TARGET_DIRECTORY_PATH`.
- When you create a image which name is like `2018-11-02-22-49-19_*.jpg` under `TARGET_DIRECTORY_PATH`, that image will be analyzed automatically.
- The shadow analysis is continued in an endless loop.
