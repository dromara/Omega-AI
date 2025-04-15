package com.omega.utils.yolo;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

/**
 * xml utils
 *
 * @author Administrator
 */
public class XmlParser {
    private static DocumentBuilderFactory dbf = null;
    private static DocumentBuilder db = null;

    public static DocumentBuilder DB() {
        if (db == null) {
            if (dbf == null) {
                dbf = DocumentBuilderFactory.newInstance();
            }
            try {
                db = dbf.newDocumentBuilder();
            } catch (ParserConfigurationException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        return db;
    }
}

