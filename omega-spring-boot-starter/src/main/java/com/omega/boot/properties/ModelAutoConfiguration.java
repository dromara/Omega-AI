package com.omega.boot.properties;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@AutoConfiguration
@EnableConfigurationProperties(ModelProperties.class)
@ConditionalOnClass(DemoService.class) // 类路径存在DemoService时生效
@ConditionalOnProperty(prefix = "model.path", name = "enabled", havingValue = "true", matchIfMissing = true)
public class ModelAutoConfiguration {
}
