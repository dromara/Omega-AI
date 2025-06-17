package com.omega.common.lib;

import java.io.File;

public class LibPaths {
    public static final String LIB_PATH = new File(LibPaths.class.getResource("/cu").getPath()).getPath() + File.separator;
}
