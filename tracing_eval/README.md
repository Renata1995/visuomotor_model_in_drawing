
#### 1. Use pull_tracing_svg_400x400.ipynb to Pull 400 x 400 size image from SVG data.
- Date will be stored in a local folder. Each tracing category has a distinct folder inside
- Filename:<age>_<session_id>_<category>.png

#### 2. Use calculate_shape_spatial.ipynb to calculate shape error and spatial error for each tracing image.
- An output file kiddraw_tracing_<iternation_name>.csv will be generated
    - Pre_Tran: Shape error before transformation
    - Post_Tran: Shape error after transformation
    - Spatio_Error are in three columns: Translation, Rotation, and Scale
- An output folder transformed_ncc is also generated to store visual transformed results