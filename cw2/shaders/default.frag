#version 450 

layout( location = 0 ) in vec3 v2fColor; 
layout( location = 1 ) in vec4 FragPos; 

layout( set = 0, binding = 0 ) uniform UScene 
{ 
	mat4 camera; 
	mat4 projection;
	mat4 projCam; 
	vec3 cameraPos;
	
} uScene; 

layout( set = 0, binding = 1 ) uniform ULight 
{ 
	vec3 light_pos[3]; 
	vec3 light_col[3];
	float numLights;
	
} uLight; 


layout( location = 0 ) out vec4 oColor; 

void main() 
{ 
	//vec3 lightPos = vec3(uLight.light_pos[0].x, uLight.light_pos[0].y, uLight.light_pos[0].z);
	//vec3 lvColor = normalize(lightPos - vec3(FragPos.x, FragPos.y, FragPos.z));
	//vec3 cvColor = normalize(uScene.cameraPos - vec3(FragPos.x, FragPos.y, FragPos.z));
	oColor = vec4( v2fColor, 1.f ); 
}
