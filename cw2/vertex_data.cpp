#include "vertex_data.hpp"

#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
namespace lut = labutils;

std::vector<ColorizedMesh> create_triangle_mesh( labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, ModelData& data )
{
	std::vector<ColorizedMesh> return_mesh;
	for (int j = 0; j < data.meshes.size(); j++) {
		// Vertex data
		std::vector<float> positions;
		std::vector<float> normals;
		std::vector<float> diffuse;
		std::vector<float> emissive;
		std::vector<float> specular;
		std::vector<float> albedo;
		std::vector<float> shiny;
		
		//Create a buffer for every light component

			for (int i = 0; i < data.meshes[j].numberOfVertices; i++) {
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].x);
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].y);
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].z);

				normals.push_back(data.vertexNormals[data.meshes[j].vertexStartIndex + i].x);
				normals.push_back(data.vertexNormals[data.meshes[j].vertexStartIndex + i].y);
				normals.push_back(data.vertexNormals[data.meshes[j].vertexStartIndex + i].z);

				diffuse.push_back(data.materials[data.meshes[j].materialIndex].diffuse.x);
				diffuse.push_back(data.materials[data.meshes[j].materialIndex].diffuse.y);
				diffuse.push_back(data.materials[data.meshes[j].materialIndex].diffuse.z);

				emissive.push_back(data.materials[data.meshes[j].materialIndex].emissive.x);
				emissive.push_back(data.materials[data.meshes[j].materialIndex].emissive.y);
				emissive.push_back(data.materials[data.meshes[j].materialIndex].emissive.z);

				specular.push_back(data.materials[data.meshes[j].materialIndex].specular.x);
				specular.push_back(data.materials[data.meshes[j].materialIndex].specular.y);
				specular.push_back(data.materials[data.meshes[j].materialIndex].specular.z);

				albedo.push_back(data.materials[data.meshes[j].materialIndex].albedo.x);
				albedo.push_back(data.materials[data.meshes[j].materialIndex].albedo.y);
				albedo.push_back(data.materials[data.meshes[j].materialIndex].albedo.z);

				shiny.push_back(data.materials[data.meshes[j].materialIndex].shininess);
				shiny.push_back(data.materials[data.meshes[j].materialIndex].metalness);
			}
		

		//printf("%d", sizeof(colors));

		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			positions.size() * sizeof(float),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			positions.size() * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

		}

		std::memcpy(posPtr, positions.data(), positions.size() * sizeof(float));
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);


		lut::Buffer vertexNormGPU = lut::create_buffer(
			aAllocator,
			normals.size() * sizeof(float),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer normStaging = lut::create_buffer(
			aAllocator,
			normals.size() * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* normPTR = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, normStaging.allocation, &normPTR); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

		}

		std::memcpy(normPTR, normals.data(), normals.size() * sizeof(float));
		vmaUnmapMemory(aAllocator.allocator, normStaging.allocation);


		lut::Buffer vertexDifGPU = lut::create_buffer(
			aAllocator,
			diffuse.size() * sizeof(float),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer difStaging = lut::create_buffer(
			aAllocator,
			diffuse.size() * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* difPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, difStaging.allocation, &difPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(difPtr, diffuse.data(), diffuse.size() * sizeof(float));
		vmaUnmapMemory(aAllocator.allocator, difStaging.allocation);


		lut::Buffer vertexEmiGPU = lut::create_buffer(
			aAllocator,
			emissive.size() * sizeof(float),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer emiStaging = lut::create_buffer(
			aAllocator,
			emissive.size() * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* emiPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, emiStaging.allocation, &emiPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(emiPtr, emissive.data(), emissive.size() * sizeof(float));
		vmaUnmapMemory(aAllocator.allocator, emiStaging.allocation);


		lut::Buffer vertexSpecGPU = lut::create_buffer(
			aAllocator,
			specular.size() * sizeof(float),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer specStaging = lut::create_buffer(
			aAllocator,
			specular.size() * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* specPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, specStaging.allocation, &specPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(specPtr, specular.data(), specular.size() * sizeof(float));
		vmaUnmapMemory(aAllocator.allocator, specStaging.allocation);


		lut::Buffer vertexAlbGPU = lut::create_buffer(
			aAllocator,
			albedo.size() * sizeof(float),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer albStaging = lut::create_buffer(
			aAllocator,
			albedo.size() * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* albPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, albStaging.allocation, &albPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(albPtr, albedo.data(), albedo.size() * sizeof(float));
		vmaUnmapMemory(aAllocator.allocator, albStaging.allocation);


		lut::Buffer vertexShiGPU = lut::create_buffer(
			aAllocator,
			shiny.size() * sizeof(float),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer shiStaging = lut::create_buffer(
			aAllocator,
			shiny.size() * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);
		
		void* shiPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, shiStaging.allocation, &shiPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}


		std::memcpy(shiPtr, shiny.data(), shiny.size() * sizeof(float));
		vmaUnmapMemory(aAllocator.allocator, shiStaging.allocation);

		lut::Fence uploadComplete = create_fence(aContext);

		lut::CommandPool uploadPool = create_command_pool(aContext);
		VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());

		}

		VkBufferCopy pcopy{};
		pcopy.size = positions.size() * sizeof(float);

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ncopy{};
		
		ncopy.size = normals.size() * sizeof(float);

		vkCmdCopyBuffer(uploadCmd, normStaging.buffer, vertexNormGPU.buffer, 1, &ncopy);

		lut::buffer_barrier(uploadCmd,
			vertexNormGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy dcopy{};

		dcopy.size = diffuse.size() * sizeof(float);

		vkCmdCopyBuffer(uploadCmd, difStaging.buffer, vertexDifGPU.buffer, 1, &dcopy);

		lut::buffer_barrier(uploadCmd,
			vertexDifGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ecopy{};

		ecopy.size = emissive.size() * sizeof(float);

		vkCmdCopyBuffer(uploadCmd, emiStaging.buffer, vertexEmiGPU.buffer, 1, &ecopy);

		lut::buffer_barrier(uploadCmd,
			vertexEmiGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy scopy{};

		scopy.size = specular.size() * sizeof(float);

		vkCmdCopyBuffer(uploadCmd, specStaging.buffer, vertexSpecGPU.buffer, 1, &scopy);

		lut::buffer_barrier(uploadCmd,
			vertexSpecGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy acopy{};

		acopy.size = albedo.size() * sizeof(float);

		vkCmdCopyBuffer(uploadCmd, albStaging.buffer, vertexAlbGPU.buffer, 1, &acopy);

		lut::buffer_barrier(uploadCmd,
			vertexAlbGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy shcopy{};

		shcopy.size = shiny.size() * sizeof(float);

		vkCmdCopyBuffer(uploadCmd, shiStaging.buffer, vertexShiGPU.buffer, 1, &shcopy);

		lut::buffer_barrier(uploadCmd,
			vertexShiGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);
		

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Submit transfer commands 
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());

		}


		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}

		
			return_mesh.push_back(ColorizedMesh{
			std::move(vertexPosGPU),
			std::move(vertexNormGPU),
			std::move(vertexDifGPU),
			std::move(vertexEmiGPU),
			std::move(vertexSpecGPU),
			std::move(vertexAlbGPU),
			std::move(vertexShiGPU),
			(unsigned int)positions.size() / 3 // three floats per position 
				});
		

	}

	return return_mesh;
	
}