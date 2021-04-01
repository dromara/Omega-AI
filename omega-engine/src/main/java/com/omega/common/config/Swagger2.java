package com.omega.common.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import springfox.documentation.builders.ApiInfoBuilder;
import springfox.documentation.builders.PathSelectors;
import springfox.documentation.builders.RequestHandlerSelectors;
import springfox.documentation.service.ApiInfo;
import springfox.documentation.spi.DocumentationType;
import springfox.documentation.spring.web.plugins.Docket;
import springfox.documentation.swagger2.annotations.EnableSwagger2;

/**
 * Swagger2配置类 @Configuration：让Spring来加载该类配置。 @EnableSwagger2：启用Swagger2。
 */
@EnableSwagger2
@Configuration
public class Swagger2 {
	
	/**
	 * 创建API应用 apiInfo() 增加API相关信息
	 * 通过select()函数返回一个ApiSelectorBuilder实例,用来控制哪些接口暴露给Swagger来展现，
	 * 本例采用指定扫描的包路径来定义指定要建立API的目录。
	 * 
	 * @return
	 */
	@Bean
	public Docket createRestApi() {
		return new Docket(DocumentationType.SWAGGER_2).apiInfo(apiInfo()).select()
				.apis(RequestHandlerSelectors.basePackage("com.talentcard.rc")).paths(PathSelectors.any()).build();
	}

	/**
	 * 创建该API的基本信息（这些基本信息会展现在文档页面中） 访问地址：http://项目实际地址/swagger-ui.html
	 * 
	 * @return
	 */
	private ApiInfo apiInfo() {
		return new ApiInfoBuilder().title("RESTful APIs").description("南海人才服务组件").termsOfServiceUrl("http://www.**.com")
				.license("证书").licenseUrl("证书地址").version("2.0").build();
	}

}
