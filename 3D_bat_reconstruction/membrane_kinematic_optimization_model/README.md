# Kinematic and membrane reconstruction.
## Flying tunnel and camera matrix.
Flying tunntl consist about 50 camera and are divided into two sections (2-4) (5-7). Svoboda method was ued to obtain the camera matrices for each section. 
<div>
 <figure  style="text-align: center;">
    <img src="images/tunnel_section_1.png"  height="330",alt="tunnel section one" />
  </figure>
  <figure  style="text-align: center;">
    <img src="images/tunnel_section_2.png"  height="330",alt="tunnel section two" />
  </figure>
</div>
Camera group one (1): 21,22,23,24,25,31,32,35,41,42,43,44,45 
 
Camera group two (2): 52,53,54,55,61,64,65,71,72,73,74,75

## Digital mesh and skeleton design
<div>
  <figure  style="text-align: center;">
    <img src="images/template_design.jpg" alt="template design" />
  </figure>
</div>

## Kinematic optimization pipeline
<div>
  <figure  style="text-align: center;">
    <img src="images/kinematic optimization network.jpg" alt="kinematic optimization" />
  </figure>
</div>

## Membrane optimization pipeline
<div>
  <figure  style="text-align: center;">
    <img src="images/membrane optimization network.jpg" alt="membrane optimization" />
  </figure>
</div>

## Kinematic update pipeline
<div>
  <figure  style="text-align: center;">
    <img src="images/kinematic update network.jpg" alt="kinematic update" />
  </figure>
</div>



---

## Kinematic reconstruction (2023)

<!-- Container: flex layout, wraps to next line if viewport shrinks -->
<div>

  <!-- Each image: width ~19% so 5 fit across with small gaps.
       aspect-ratio:1/1 keeps it square, object-fit:cover crops to fill.
       Replace src="..." with your image URL (e.g., images/pic1.png or raw githubusercontent link).
       Wrap with <a> to make the image clickable (optional). -->

  <figure  style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_14_1/Brunei_2023_bat_14_1_smoothed_trajectory.gif"  width="330" height="330" alt="Brunei_2023_bat_14_1" />
  </figure>

  <figure style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_15_1/Brunei_2023_bat_15_1_smoothed_trajectory.gif"  width="330" height="330" alt="Brunei_2023_bat_15_1" />
  </figure>

  <figure style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_15_2/Brunei_2023_bat_15_2_smoothed_trajectory.gif"  width="330" height="330" alt="Brunei_2023_bat_15_2" />
  </figure>

  <figure style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_16/Brunei_2023_bat_16_trajectory.gif"      width="330" height="330" alt="Brunei_2023_bat_16" />
  </figure>

  <figure style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_test_13_1/Brunei_2023_bat_test_13_1_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2023_bat_test_13_1"/>
  </figure>

  <figure  style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_test_13_2/Brunei_2023_bat_test_13_2_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2023_bat_test_13_2"/>
  </figure>

  <figure  style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_test_15_1/Brunei_2023_bat_test_15_1_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2023_bat_test_15_1"/>
  </figure>

  <figure style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_test_16_1/Brunei_2023_bat_test_16_1_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2023_bat_test_16_1"/>
  </figure>

  <figure  style="text-align: center;">
    <img src="drivers/2023/result_plot/Brunei_2023_bat_test_17_1/Brunei_2023_bat_test_17_1_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2023_bat_test_17_1"/>
  </figure>
</div>

## Kinematic reconstruction (2024)
### HIPCER 
#### 2-4 (first half of tunnel)
<div style="display: flex; flex-wrap: wrap; gap: 10px; max-width: 900px;">
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER019_FlightTest1_2_4/Brunei_2024_HIPCER019_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER019_FlightTest1_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER021_FlightTest1_2_4/Brunei_2024_HIPCER021_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER021_FlightTest1_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER022_FlightTest1_2_4/Brunei_2024_HIPCER022_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER022_FlightTest1_2_4"/>
    </figure>
        <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER022_FlightTest1_2_4_sec/Brunei_2024_HIPCER022_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER022_FlightTest1_2_4_sec"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER023_FlightTest3_2_4/Brunei_2024_HIPCER023_FlightTest3_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER023_FlightTest3_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER023_FlightTest4_2_4/Brunei_2024_HIPCER023_FlightTest4_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER023_FlightTest4_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER024_FlightTest1_2_4/Brunei_2024_HIPCER024_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER024_FlightTest1_2_4"/>
    </figure>
        <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER024_FlightTest2_2_4/Brunei_2024_HIPCER024_FlightTest2_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER024_FlightTest2_2_4"/>
    </figure>
    </figure>
        <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER025_FlightTest1_2_4/Brunei_2024_HIPCER025_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER025_FlightTest1_2_4"/>
    </figure>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER025_FlightTest2_2_4/Brunei_2024_HIPCER025_FlightTest2_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER025_FlightTest2_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER025_FlightTest2_2_4_sec/Brunei_2024_HIPCER025_FlightTest2_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER025_FlightTest2_2_4_sec"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER025_FlightTest3_2_4/Brunei_2024_HIPCER025_FlightTest3_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER025_FlightTest3_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPCER025_FlightTest3_2_4_sec/Brunei_2024_HIPCER025_FlightTest3_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPCER025_FlightTest3_2_4_sec"/>
    </figure>
</div>

#### 5-7 (second half of tunnel)

### HIPDYA
#### 2-4 (first half of tunnel)
<div>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPDYA001_FlightTest2_2_4/Brunei_2024_HIPDYA001_FlightTest2_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPDYA001_FlightTest2_2_4"/>
    </figure>
</div>

#### 5-7 (second half of tunnel)
<div>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_HIPDYA001_FlightTest3_5_7/Brunei_2024_HIPDYA001_FlightTest3_5_7_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_HIPDYA001_FlightTest3_5_7"/>
    </figure>
</div>

### RHIBOR
#### 2-4 (first half of tunnel)
<div>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHIBOR001_FlightTest2_2_4/Brunei_2024_RHIBOR001_FlightTest2_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHIBOR001_FlightTest2_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHIBOR001_FlightTest3_2_4/Brunei_2024_RHIBOR001_FlightTest3_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHIBOR001_FlightTest3_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHIBOR002_FlightTest1_2_4/Brunei_2024_RHIBOR002_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHIBOR002_FlightTest1_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHIBOR002_FlightTest2_2_4/Brunei_2024_RHIBOR002_FlightTest2_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHIBOR002_FlightTest2_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHIBOR002_FlightTest2_2_4_sec/Brunei_2024_RHIBOR002_FlightTest2_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHIBOR002_FlightTest2_2_4_sec"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHIBOR002_FlightTest3_2_4/Brunei_2024_RHIBOR002_FlightTest3_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHIBOR002_FlightTest3_2_4"/>
    </figure>
</div>

#### 5-7 (second half of tunnel)

### RHITRT
#### 2-4 (first half of tunnel)
<div>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHITRI002_FlightTest1_2_4/Brunei_2024_RHITRI002_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHITRI002_FlightTest1_2_4"/>
    </figure>
</div>

#### 5-7 (second half of tunnel)

### RHISED
#### 2-4 (first half of tunnel)
<div>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHISED001_FlightTest3_2_4/Brunei_2024_RHISED001_FlightTest3_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHISED001_FlightTest3_2_4"/>
    </figure>
        <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHISED002_FlightTest3_2_4/Brunei_2024_RHISED002_FlightTest3_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHISED002_FlightTest3_2_4"/>
    </figure>
    <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHISED003_FlightTest1_2_4/Brunei_2024_RHISED003_FlightTest1_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHISED003_FlightTest1_2_4"/>
    </figure>
     <figure  style="text-align: center;">
        <img src="drivers/2024/result_plot/Brunei_2024_RHISED003_FlightTest3_2_4/Brunei_2024_RHISED003_FlightTest3_2_4_smoothed_trajectory.gif" width="330" height="330" alt="Brunei_2024_RHISED003_FlightTest3_2_4"/>
    </figure>
</div>

#### 5-7 (second half of tunnel)
