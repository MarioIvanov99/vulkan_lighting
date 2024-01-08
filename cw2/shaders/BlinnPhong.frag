#version 450 

layout( location = 0 ) in vec4 FragPos;
layout( location = 1 ) in vec3 fNormals;
layout( location = 2 ) in vec3 fDiffuse;
layout( location = 3 ) in vec3 fSpecular; 
layout( location = 4 ) in vec3 fEmissive;
layout( location = 5 ) in float fShiny; 

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
	vec3 lightPos = vec3(uLight.light_pos[0].x, uLight.light_pos[0].y, uLight.light_pos[0].z);
	vec3 lightCol = vec3(uLight.light_col[0].x, uLight.light_col[0].y, uLight.light_col[0].z);
	vec3 ambient = vec3(0.02, 0.02, 0.02);
	vec3 lDir = normalize(lightPos - vec3(FragPos.x, FragPos.y, FragPos.z));
	vec3 cDir = normalize(uScene.cameraPos - vec3(FragPos.x, FragPos.y, FragPos.z));
	vec3 hVector = normalize((lDir+cDir)/2.0);
	vec3 normal = normalize(fNormals);
	
	vec3 blinnColor = fEmissive + ambient*fDiffuse +(fDiffuse/3.1415926535897932384626433832795 + (fShiny+2.0)/8.0*max(pow(dot(normal, hVector), fShiny), 0)*fSpecular)*max(dot(normal, lDir), 0)*lightCol;
	
    oColor = vec4(blinnColor, 1.0);
}
