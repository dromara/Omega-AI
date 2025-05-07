package com.omega.boot.properties;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "model")
public class ModelProperties {

    private String name;
    private String path;



}
