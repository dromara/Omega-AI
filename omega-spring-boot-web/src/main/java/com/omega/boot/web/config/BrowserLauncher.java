package com.omega.boot.web.config;

import org.slf4j.Logger;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.awt.*;
import java.net.URI;

/**
 * 在云服务器上能打开访问
 */
@Component
public class BrowserLauncher {

    @Value("${server.port:8080}")
    private int port;

    private Logger logger = org.slf4j.LoggerFactory.getLogger(BrowserLauncher.class);

    @EventListener(ApplicationReadyEvent.class)
    public void openBrowserAfterStartup() {

        try {
            String url = "http://0.0.0.0:" + port; // 你的应用地址
            logger.info("正在打开浏览器: " + url);

            if (Desktop.isDesktopSupported()) {
                Desktop desktop = Desktop.getDesktop();
                if (desktop.isSupported(Desktop.Action.BROWSE)) {
                    desktop.browse(new URI(url));
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
