package com.omega.example.yolo.test.deepsort;

/**
 * 
 */
public class Rect {
    public final int x, y, width, height;
    
    public Rect(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
    
    @Override
    public String toString() {
        return String.format("BBox[x=%d, y=%d, w=%d, h=%d]", x, y, width, height);
    }

    /**
     * 面积
     * @return
     */
	public int area() {
		return width * height;
	}
}