package com.omega.boot.starter;

import cn.hutool.json.JSONObject;
import com.omega.boot.starter.entity.ModelData;
import com.omega.boot.starter.utils.JsonUtils;
import com.omega.boot.starter.utils.PackageScannerUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.util.StringUtils;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 模型配置文件初始化加载器
 *
 * @author haylee
 * @date 2025/05/14 14:33
 */

public class ModelApplicationInitializer implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    private static final Logger logger = LoggerFactory.getLogger(ModelApplicationInitializer.class);
    private static final String MODEL_PATH_PROPERTY = "model.path";
    private static final String MODEL_CONFIG_BEAN_NAME = "modelConfig";
    private static final String CONFIG = "config.json";
    private static final String TOKENIZER_CONFIG = "tokenizer_config.json";
    private static final String MODEL_TYLE = "model_type";

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        ConfigurableEnvironment env = applicationContext.getEnvironment();
        Binder binder = Binder.get(env);

        List<String> paths = binder.bind(MODEL_PATH_PROPERTY, Bindable.listOf(String.class))
                .orElse(Collections.emptyList());

        if (paths.isEmpty()) {
            logger.debug("No model paths configured");
            return;
        }

        logger.info("Configured model paths: {}", paths);

        Map<String, ModelData> modelDataMap = new HashMap<>();
        for (String path : paths) {
            try {
                String configUrl = path + File.separator + CONFIG;
                JSONObject config = JsonUtils.readJson(configUrl);
                if(config == null){
                    logger.error("read model data is empty from configUrl: {}", path);
                    continue;
                }
                String modelType = config.getStr(MODEL_TYLE);
                if(StringUtils.isEmpty(modelType)){
                    logger.error("read model model_type is empty from configUrl: {}", path);
                    continue;
                }
                String tokenizerConfigUrl = path + File.separator + TOKENIZER_CONFIG;
                JSONObject tokenizerConfig = JsonUtils.readJson(tokenizerConfigUrl);
                if(tokenizerConfig == null){
                    logger.error("read tokenizer_config is empty from configUrl: {}", tokenizerConfigUrl);
                    continue;
                }
                String tokenizerClass = tokenizerConfig.getStr("tokenizer_class");
                if(StringUtils.isEmpty(tokenizerClass)){
                    logger.error("read tokenizer_class is empty from configUrl: {}", tokenizerConfigUrl);
                    continue;
                }

                ModelData modelData = new ModelData(path, config, tokenizerConfig);
                modelDataMap.put(modelType, modelData);
            } catch (Exception e) {
                logger.error("Failed to read model config from path: {}", path, e);
            }
        }

        //注册bean
        applicationContext.getBeanFactory().registerSingleton(
                MODEL_CONFIG_BEAN_NAME,
                modelDataMap
        );

    }
}
