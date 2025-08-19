import numpy as np
import sys

def main():
    #   
    if len(sys.argv) != 7:
        print("    USE: python depth_map.py <width>x<height> <points_file> fx fy cx cy")
        print("Example: python depth_map.py 640x480 points.txt 525.0 525.0 320.0 240.0")
        sys.exit(1)
    
    #  
    resolution = sys.argv[1].split('x')
    width, height = int(resolution[0]), int(resolution[1])
    points_file = sys.argv[2]
    
    #   
    fx = float(sys.argv[3])
    fy = float(sys.argv[4])
    cx = float(sys.argv[5])
    cy = float(sys.argv[6])
    
    #    K (3x3)
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    print(f"  :\n{K}")
    
    #   
    points = []
    with open(points_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    #   
                    x, y, z, r = map(float, parts[:4])
                    points.append((x, y, z, r))
                except ValueError:
                    continue  #   
    if not points:
        print(":      ")
        sys.exit(1)
        
    print(f" : {len(points)}")
    
    #    (   )
    depth_map = np.full((height, width), np.inf)
    
    #  
    for i, pt in enumerate(points):
        xpt, ypt, zpt, r = pt
        
        z = xpt
        x = ypt
        y = zpt
        r = z
        #    
        if z <= 0 or z>5000:
            continue
        
        #    
        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
        
        #    
        u_idx = int(round(u))
        v_idx = int(round(v))
        
        #   
        if 0 <= u_idx < width and 0 <= v_idx < height:
            #    ( )
            if r < depth_map[v_idx, u_idx]:
                depth_map[v_idx, u_idx] = r
    
    #    0 ()
    depth_map[np.isinf(depth_map)] = 0
    
    #   
    depth_visual = np.uint8(255 * depth_map / np.max(depth_map))
    
    #  
    npy_file = points_file.rsplit('.', 1)[0] + "_depth.npy"
    np.save(npy_file, depth_map)
    
    png_file = points_file.rsplit('.', 1)[0] + "_depth.png"
    from PIL import Image
    Image.fromarray(depth_visual).save(png_file)
    
    print(f"     numpy: {npy_file}")
    print(f"  : {png_file}")
    print(f" : {width}x{height}")
    print(f" : {np.min(depth_map[depth_map>0]):.2f} - {np.max(depth_map):.2f}")

if __name__ == "__main__":
    main()
