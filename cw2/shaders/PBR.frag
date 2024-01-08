#version 450 

layout( location = 0 ) in vec4 FragPos;
layout( location = 1 ) in vec3 fNormals;
layout( location = 2 ) in vec3 fAlbedo;
layout( location = 3 ) in vec3 fEmissive;
layout( location = 4 ) in float fShiny; 
layout( location = 5 ) in float fMetal; 

layout( set = 0, binding = 0 ) uniform UScene 
{ 
	mat4 camera; 
	mat4 projection;
	mat4 projCam; 
	vec3 cameraPos;
	
} uScene; 

layout( set = 0, binding = 1 ) uniform ULight 
{ 
	vec4 light_pos[3]; 
	vec4 light_col[3];
	float numLights;
	
} uLight; 


layout( location = 0 ) out vec4 oColor; 

void main() 
{ 
	vec3 LightSum = vec3(0.0, 0.0, 0.0);
	float pi = 3.1415926535897932384626433832795;
	vec3 ambient = vec3(0.02, 0.02, 0.02);
	vec3 normal = normalize(fNormals);
	vec3 f0 = (1.0-fMetal)*vec3(0.04, 0.04, 0.04) + fMetal*fAlbedo;
	vec3 ambientL = ambient*fAlbedo;

	for(int i =0; i < int(uLight.numLights); i++){
		vec3 lightPos = vec3(uLight.light_pos[i].x, uLight.light_pos[i].y, uLight.light_pos[i].z);
		vec3 lightCol = vec3(uLight.light_col[i].x, uLight.light_col[i].y, uLight.light_col[i].z);

		vec3 lDir = normalize(lightPos - vec3(FragPos.x, FragPos.y, FragPos.z));
		vec3 cDir = normalize(uScene.cameraPos - vec3(FragPos.x, FragPos.y, FragPos.z));
		vec3 hVector = normalize((lDir+cDir)/2.0);
	
		vec3 fresnel = f0 + (vec3(1.0, 1.0, 1.0)-f0)*(1-pow(dot(hVector, cDir), 5));
		//vec3 diffuse = (fAlbedo/pi)*(vec3(1.0, 1.0, 1.0)-fresnel)*(1.0-fMetal);	
		float f90 = 0.5+2.0*fMetal*dot(lDir, hVector)*dot(lDir, hVector);
		vec3 diffuse = fAlbedo/pi * (1.0 + (f90 - 1.0) * pow(1.0 - dot(normal, lDir), 5.0)) * (1.0 + (f90 - 1.0) * pow(1.0 - dot(normal, cDir), 5.0));

		float distribution = (fShiny+2.0)/(2.0*pi)*max(pow(dot(normal, hVector), fShiny), 0);


		float innermin = min(2.0*(max(dot(normal, hVector), 0.0)*max(dot(normal, cDir), 0.0))/dot(cDir, hVector), 2.0*(max(dot(normal, hVector), 0.0)*max(dot(normal, lDir), 0.0))/dot(cDir, hVector));
		float masking = min(1.0, innermin);
	
		vec3 fr = diffuse + distribution*fresnel*masking / (4.0*max(dot(normal, cDir), 0.0001)*max(dot(normal, lDir), 0.0001));

		LightSum = LightSum +  fr*lightCol*max(dot(normal, lDir), 0);
	}
	
	vec3 pbrColor = fEmissive + ambientL + LightSum;
	oColor = vec4(pbrColor, 1.0);
}
