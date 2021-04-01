package com.omega.common.utils;

import java.util.Random;

/**
 * 数学工具
 * @author Administrator
 *
 */
public class MathUtils {
	
	public static Random random;
	
	/**
	 * 整形随机数(范围)
	 * @param start
	 * @param end
	 * @return
	 */
	public static int randomInt(int end) {
		return MathUtils.getRandom().nextInt(end);
	}
	
	/**
	 * 整形随机数(范围)
	 * @param end
	 * @param size
	 * @return
	 */
	public static int[] randomInt(int end,int size) {
		int[] result = new int[size];
		for(int i = 0;i<size;i++) {
			result[i] = MathUtils.getRandom().nextInt(end);
		}
		return result;
	}
	
	/**
	 * 整形随机数不重复(范围)
	 * @param end
	 * @param size
	 * @return
	 */
	public static int[] randomIntNotRepeat(int end,int size) {
		int[] result = new int[size];
		for(int i = 0;i<size;i++) {
			int temp = MathUtils.getRandom().nextInt(end);
			MathUtils.randomIntNotRepeat(result, i, temp, end);
		}
		return result;
	}
	
	public static void randomIntNotRepeat(int[] data,int index,int value,int end) {
		for(int i = 0;i<data.length;i++) {
			if(!MathUtils.isExist(data, value)) {
				data[index] = value;
				break;
			}else {
				int temp = MathUtils.getRandom().nextInt(end);
				MathUtils.randomIntNotRepeat(data, index, temp, end);
			}
		}
	}
	
	public static boolean isExist(int[] data,int value) {
		for(int temp:data) {
			if(temp == value) {
				return true;
			}
		}
		return false;
	}
	
	public static Random getRandom() {
		if(random == null) {
			random = new Random();
		}
		return random;
	}
	
}
