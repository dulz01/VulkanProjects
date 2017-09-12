// Vulkan header integrated with GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS // makes sure that functions use radians
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // changes the depth range start from -1.0 to 0.0
#include <glm.hpp>
#include <gtc/matrix_transform.hpp> // functions that can generate model transformations
#include <gtx/hash.hpp> // for hash function converting user types into a hash key for std::map

#define STB_IMAGE_IMPLEMENTATION // includes the function bodies from stb_image.h
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION // includes the function bodies from tiny_obj_loader.h 
#include <tiny_obj_loader.h>

#include <openvr.h>

// Reporting and propagating errors
#include <stdexcept>
#include <iostream>

#include <cstring> // for strcmp function
#include <algorithm> // for max and min functions
#include <fstream> // for file reading
#include <chrono> // timekeeping
#include <vector>
#include <array>
#include <set>
#include <unordered_map>

const int WIDTH = 640;
const int HEIGHT = 320;

// path to model and texture
const std::string MODEL_PATH = "models/chalet.obj";
const std::string TEXTURE_PATH = "textures/chalet.jpg";

//-----------------------------------------------------------------------------
// Purpose: shows which validation layers are required
//-----------------------------------------------------------------------------
// "VK_LAYER_LUNARG_standard_validation"

// "VK_LAYER_GOOGLE_threading",
// "VK_LAYER_LUNARG_parameter_validation",
// "VK_LAYER_LUNARG_object_tracker",
// "VK_LAYER_LUNARG_core_validation",
// "VK_LAYER_LUNARG_image" not available
// "VK_LAYER_LUNARG_swapchain" not available

// TODO: Learn how to install more validation layers

const std::vector<const char*> validation_layers = {
  "VK_LAYER_LUNARG_standard_validation"
};

//-----------------------------------------------------------------------------
// Purpose: shows required device extensions
//----------------------------------------------------------------------------- 
const std::vector<const char*> device_extensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//-----------------------------------------------------------------------------
// Purpose: activate validation layers depending on configuration mode
//----------------------------------------------------------------------------- 
#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

//-----------------------------------------------------------------------------
// Purpose: looking up the address of the extension to be exclusively loaded
//-----------------------------------------------------------------------------
VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo,
  const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback)
{
  // returns a nullptr if the extension function is not loaded
  auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pCallback);
  }
  else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

//-----------------------------------------------------------------------------
// Note: this function has to be either static or outside the class in order...
// ...to be called in cleanup
//-----------------------------------------------------------------------------
void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
  if (func != nullptr) {
    func(instance, callback, pAllocator);
  }
}

//-----------------------------------------------------------------------------
// Purpose: setting the types of queue families we should look for
//-----------------------------------------------------------------------------
struct QueueFamilyIndices {
  // -1 denotes "not found"
  int graphics_family = -1; 
  int present_family = -1;

  bool isComplete() {
    return graphics_family >= 0 && present_family >= 0;
  }
};

//-----------------------------------------------------------------------------
// Purpose: holds the properties a swap chain needs to be created
//-----------------------------------------------------------------------------
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

//-----------------------------------------------------------------------------
// Purpose: holds the vertices data
//-----------------------------------------------------------------------------
struct Vertex {
  glm::vec3 pos;
  glm::vec3 colour;
  glm::vec2 tex_coord;

  // passing the data format to the vertex shader
  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription binding_description = {};
    binding_description.binding = 0;
    binding_description.stride = sizeof(Vertex);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return binding_description;
  }

  // describes how to handle vertex input
  static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 3> attribute_descriptions = {};

    attribute_descriptions[0].binding = 0;
    attribute_descriptions[0].location = 0;
    attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[0].offset = offsetof(Vertex, pos);

    attribute_descriptions[1].binding = 0;
    attribute_descriptions[1].location = 1;
    attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[1].offset = offsetof(Vertex, colour);

    attribute_descriptions[2].binding = 0;
    attribute_descriptions[2].location = 2;
    attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attribute_descriptions[2].offset = offsetof(Vertex, tex_coord);

    return attribute_descriptions;
  }

  // overriding == for std::map check
  bool operator==(const Vertex& other) const {
    return pos == other.pos && colour == other.colour && tex_coord == other.tex_coord;
  }
};

//-----------------------------------------------------------------------------
// Purpose: converting Vertex into a hash to be used as a key in std::map
//-----------------------------------------------------------------------------
namespace std {
  template<> struct hash<Vertex> {
    size_t operator()(Vertex const& vertex) const {
      return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.colour) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.tex_coord) << 1);
    }
  };
}

//-----------------------------------------------------------------------------
// Purpose: data we want the vertex shader to have
//-----------------------------------------------------------------------------
struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class HelloTriangleApplication {
public:
  //---------------------------------------------------------------------------
  // Purpose: run all the important functions
  //---------------------------------------------------------------------------
  void run() {
    initWindow();
    initVulkan();
    initVRCompositor();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow* companion_window_;

  VkInstance instance_;
  VkDebugReportCallbackEXT callback_;
  VkSurfaceKHR surface_;
    
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE; // implicitly destroyed with VkInstance
  VkDevice device_;

  VkQueue graphics_queue_;
  VkQueue present_queue_;

  VkSwapchainKHR swap_chain_;
  std::vector<VkImage> swap_chain_images_; // implicitly created and destroyed
  VkFormat swap_chain_image_format_;
  VkExtent2D swap_chain_extent_;
  std::vector<VkImageView> swap_chain_image_views_;
  std::vector<VkFramebuffer> swap_chain_framebuffers_;

  VkRenderPass render_pass_;
  VkDescriptorSetLayout descriptor_set_layout_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;

  VkCommandPool command_pool_;

  VkImage depth_image_;
  VkDeviceMemory depth_image_memory_;
  VkImageView depth_image_view_;

  VkImage texture_image_;
  VkDeviceMemory texture_image_memory_;
  VkImageView texture_image_view_;
  VkSampler texture_sampler_;

  std::vector<Vertex> vertices_;
  std::vector<uint32_t> indices_;
  VkBuffer vertex_buffer_;
  VkDeviceMemory vertex_buffer_memory_;
  VkBuffer index_buffer_;
  VkDeviceMemory index_buffer_memory_;

  VkBuffer uniform_buffer_;
  VkDeviceMemory uniform_buffer_memory_;

  VkDescriptorPool descriptor_pool_;
  VkDescriptorSet descriptor_set_;

  std::vector<VkCommandBuffer> command_buffers_;

  VkSemaphore image_available_semaphore_;
  VkSemaphore render_finished_semaphore_;

  //------------------//
  // variables for VR //
  //------------------//
  vr::IVRSystem *p_hmd_;
  vr::IVRRenderModels *p_render_models_;

  struct FramebufferDesc {
    VkImage image;
    VkImageLayout image_layout;
    VkDeviceMemory device_memory;
    VkImageView image_view;
    VkImage depth_stencil_image;
    VkImageLayout depth_stencil_image_layout;
    VkDeviceMemory depth_stencil_device_memory;
    VkImageView depth_stencil_image_view;
    VkRenderPass render_pass;
    VkFramebuffer frame_buffer;
  };
  FramebufferDesc left_eye_desc_;
  FramebufferDesc right_eye_desc_;

  VkBuffer scene_uniform_buffer_[2];
  VkDeviceMemory scene_uniform_buffer_memory_[2];
  
  glm::mat4 mat4_proj_left_;
  glm::mat4 mat4_proj_right_;
  glm::mat4 mat4_eye_pos_left_;
  glm::mat4 mat4_eye_pos_right_;

  //---------------------------------------------------------------------------
  // Purpose: initialise the windowing system
  //---------------------------------------------------------------------------
  void initWindow() {
    glfwInit(); // Initialize GLFW

    // GLFW_NO_API tells GLFW not to create an OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); 
    
    //---------//
    // VR code //
    //---------//
    memset(&left_eye_desc_, 0, sizeof(left_eye_desc_));
    memset(&right_eye_desc_, 0, sizeof(right_eye_desc_));

    vr::EVRInitError e_error = vr::VRInitError_None;
    p_hmd_ = vr::VR_Init(&e_error, vr::VRApplication_Scene);
    if (e_error != vr::VRInitError_None) {
      p_hmd_ = NULL;
      throw std::runtime_error("VR_Init Failed");
    }

    p_render_models_ = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &e_error);
    if (!p_render_models_) {
      p_hmd_ = NULL;
      vr::VR_Shutdown();
      throw std::runtime_error("Unable to get render model interface.");
    }
    //----------------//
    // End of VR code //
    //----------------//

    // store a reference to the window when creating it
    companion_window_ = glfwCreateWindow(WIDTH, HEIGHT, "VulkanVR", nullptr, nullptr);

    glfwSetWindowUserPointer(companion_window_, this);
    glfwSetWindowSizeCallback(companion_window_, HelloTriangleApplication::onWindowResized);
  }

  //---------------------------------------------------------------------------
  // Purpose: calls all the functions required to initialise Vulkan
  //---------------------------------------------------------------------------
  void initVulkan() {
    // setup
    createInstance();
    setupDebugCallback();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();

    // presentation
    createSwapChain();
    createImageViews(); // must be after creating the swap chain

    // graphics pipeline
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();

    // drawing
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffer();
    //createPerEyeUniformBuffer();

    createDescriptorPool();
    createDescriptorSet();
    createCommandBuffers();
    createSemaphores();
  }

  //---------------------------------------------------------------------------
  // Purpose: Initialize the VR custom compositor
  //---------------------------------------------------------------------------
  void initVRCompositor() {
    vr::EVRInitError error = vr::VRInitError_None;
    
    if (!vr::VRCompositor()) {
      throw std::runtime_error("Compositor initialization failed.");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: 
  //---------------------------------------------------------------------------
  void mainLoop() {
    // keeping the application running until an error...
    // ... or the window is closed
    while (!glfwWindowShouldClose(companion_window_)) {
      glfwPollEvents();

      updateUniformBuffer();
      drawFrame();
    }

    vkDeviceWaitIdle(device_);
  }

  //---------------------------------------------------------------------------
  // Purpose: Cleaning up objects related to the swap chain
  //---------------------------------------------------------------------------
  void cleanupSwapChain() {
    vkDestroyImageView(device_, depth_image_view_, nullptr);
    vkDestroyImage(device_, depth_image_, nullptr);
    vkFreeMemory(device_, depth_image_memory_, nullptr);

    // delete before image views and render pass after rendering finishes
    for (size_t i = 0; i < swap_chain_framebuffers_.size(); i++) {
      vkDestroyFramebuffer(device_, swap_chain_framebuffers_[i], nullptr);
    }

    // free up the command buffers so we can reuse them
    vkFreeCommandBuffers(device_, command_pool_, static_cast<uint32_t>(command_buffers_.size()), command_buffers_.data());

    vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
    vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    vkDestroyRenderPass(device_, render_pass_, nullptr);

    for (size_t i = 0; i < swap_chain_image_views_.size(); i++) {
      vkDestroyImageView(device_, swap_chain_image_views_[i], nullptr);
    }

    vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
  }

  //---------------------------------------------------------------------------
  // Purpose: Destroying all the Vulkan objects explicitly created by us
  //---------------------------------------------------------------------------
  void cleanup() {
    //---------//
    // VR code //
    //---------//
    if (p_hmd_) {
      vr::VR_Shutdown();
      p_hmd_ = NULL;
    }

    FramebufferDesc *pFramebufferDescs[2] = { &left_eye_desc_, &right_eye_desc_ };
    for (int32_t i = 0; i < 2; i++) {
      if (pFramebufferDescs[i]->image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, pFramebufferDescs[i]->image_view, nullptr);
        vkDestroyImage(device_, pFramebufferDescs[i]->image, nullptr);
        vkFreeMemory(device_, pFramebufferDescs[i]->device_memory, nullptr);
        vkDestroyImageView(device_, pFramebufferDescs[i]->depth_stencil_image_view, nullptr);
        vkDestroyImage(device_, pFramebufferDescs[i]->depth_stencil_image, nullptr);
        vkFreeMemory(device_, pFramebufferDescs[i]->depth_stencil_device_memory, nullptr);
        vkDestroyRenderPass(device_, pFramebufferDescs[i]->render_pass, nullptr);
        vkDestroyFramebuffer(device_, pFramebufferDescs[i]->frame_buffer, nullptr);
      }
    }

    for (uint32_t eye = 0; eye < 2; eye++) {
      if (scene_uniform_buffer_[eye] != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, scene_uniform_buffer_[eye], nullptr);
        scene_uniform_buffer_[eye] = VK_NULL_HANDLE;
      }

      if (scene_uniform_buffer_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, scene_uniform_buffer_memory_[eye], nullptr);
        scene_uniform_buffer_memory_[eye] = VK_NULL_HANDLE;
      }
    }
    //----------------//
    // End of VR code //
    //----------------//

    cleanupSwapChain(); // must be done before device

    vkDestroySampler(device_, texture_sampler_, nullptr);
    vkDestroyImageView(device_, texture_image_view_, nullptr);

    vkDestroyImage(device_, texture_image_, nullptr);
    vkFreeMemory(device_, texture_image_memory_, nullptr);

    vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);

    vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
    vkDestroyBuffer(device_, uniform_buffer_, nullptr);
    vkFreeMemory(device_, uniform_buffer_memory_, nullptr);

    vkDestroyBuffer(device_, index_buffer_, nullptr);
    vkFreeMemory(device_, index_buffer_memory_, nullptr);

    vkDestroyBuffer(device_, vertex_buffer_, nullptr);
    vkFreeMemory(device_, vertex_buffer_memory_, nullptr);

    vkDestroySemaphore(device_, render_finished_semaphore_, nullptr);
    vkDestroySemaphore(device_, image_available_semaphore_, nullptr);

    vkDestroyCommandPool(device_, command_pool_, nullptr);

    vkDestroyDevice(device_, nullptr);
    DestroyDebugReportCallbackEXT(instance_, callback_, nullptr);

    vkDestroySurfaceKHR(instance_, surface_, nullptr); // must be destroyed before the instance
    vkDestroyInstance(instance_, nullptr); // must be destroyed right before the program exits

    // clean up resources and terminating GLFW
    glfwDestroyWindow(companion_window_);
    glfwTerminate();
  }

  //---------------------------------------------------------------------------
  // Purpose: resize the window and recreate the swap chain to fit
  //---------------------------------------------------------------------------
  static void onWindowResized(GLFWwindow* window, int width, int height) {
    if (width == 0 || height == 0) { return; }

    HelloTriangleApplication* app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    app->recreateSwapChain();
  }

  //---------------------------------------------------------------------------
  // Purpose: to recreate the swap chain when you for example, resize the window
  //---------------------------------------------------------------------------
  void recreateSwapChain() {
    vkDeviceWaitIdle(device_); // wait until resources aren't in use

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createDepthResources();
    createFramebuffers();
    createCommandBuffers();
  }
  
  //---------------------------------------------------------------------------
  // Purpose: initializing Vulkan with an instance
  //---------------------------------------------------------------------------
  void createInstance() {
    // checking if validation layers are available when enabled
    if (enable_validation_layers && !checkValidationLayerSupport()) {
      throw std::runtime_error("Validation layers requested, but not available!");
    }

    // optional data
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "VulkanVR";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    // required data
    // tells Vulkan which global extensions and validations to use
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    auto extensions = getRequiredExtensions();
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    // included validation layer data into the struct if enabled
    if (enable_validation_layers) {
      create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
      create_info.ppEnabledLayerNames = validation_layers.data();
    }
    else {
      create_info.enabledLayerCount = 0;
    }

    // *first param: pointer to struct with creation info
    // *second param: pointer to custom allocator callbacks
    // *third param: pointer to the variable that stores the handle to the new object
    if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create instance!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: telling Vulkan about the callback function
  //---------------------------------------------------------------------------
  void setupDebugCallback() {
    if (!enable_validation_layers) { return; }
    
    VkDebugReportCallbackCreateInfoEXT create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;

    // allows you to filter what type of messages you would like to receive
    create_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;

    // specifies the pointer to the callback function
    create_info.pfnCallback = debugCallback;

    if (CreateDebugReportCallbackEXT(instance_, &create_info, nullptr, &callback_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to set up debug callback!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: create a window surface using GLFW's function
  //---------------------------------------------------------------------------
  void createSurface() {
    if (glfwCreateWindowSurface(instance_, companion_window_, nullptr, &surface_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create window surface!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: find a graphics card and check if it supports any Vulkan features
  //---------------------------------------------------------------------------
  void pickPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

    // no point going further if there's no devices with Vulkan support
    if (device_count == 0) {
      throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    // allocates an array to hold all of the VkPhysicalDevice handles
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    // checks if the physical devices meet any requirements set in isDeviceSuitable()
    for (const auto& device : devices) {
      if (isDeviceSuitable(device)) {
        physical_device_ = device;
        break;
      }
    }

    if (physical_device_ == VK_NULL_HANDLE) {
      throw std::runtime_error("Failed to find a suitable GPU!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: sets up a logical device to interface with the physical device
  //---------------------------------------------------------------------------
  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physical_device_);

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set<int> unique_queue_families = { indices.graphics_family, indices.present_family };

    // Vulkan lets you assign priorities to queues to influence scheduling...
    // ...of the command buffer execution.
    // between floating point numbers 0.0 and 1.0
    // required, even if there's a single queue
    float queue_priority = 1.0f;
    for (int queue_family : unique_queue_families) {
      // this struct describes the number of queues we want for a single queue family
      VkDeviceQueueCreateInfo queue_create_info = {};
      queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queue_create_info.queueFamilyIndex = queue_family;
      queue_create_info.queueCount = 1;
      queue_create_info.pQueuePriorities = &queue_priority;
      queue_create_infos.push_back(queue_create_info);
    }
    
    // specifying what set of device features we'll be using
    VkPhysicalDeviceFeatures device_features = {};
    device_features.samplerAnisotropy = VK_TRUE;

    // creating the logical device
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    // pointers to the queue create info
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();

    // pointer to the device feature
    create_info.pEnabledFeatures = &device_features;

    // extensions and validation layers
    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();

    if (enable_validation_layers) {
      create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
      create_info.ppEnabledLayerNames = validation_layers.data();
    }
    else {
      create_info.enabledLayerCount = 0;
    }

    // instantiating the logical device with specified info
    if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create logical device!");
    }

    // retrieving the queue handles for each queue family
    vkGetDeviceQueue(device_, indices.graphics_family, 0, &graphics_queue_);
    vkGetDeviceQueue(device_, indices.present_family, 0, &present_queue_);
  }

  //---------------------------------------------------------------------------
  // Purpose: create the swap chain
  //---------------------------------------------------------------------------
  void createSwapChain() {
    SwapChainSupportDetails swap_chain_support = querySwapChainSupport(physical_device_);

    // getting the three required pieces of information from helper functions
    VkSurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
    VkPresentModeKHR present_mode = chooseSwapPresentMode(swap_chain_support.present_modes);
    VkExtent2D extent = chooseSwapExtent(swap_chain_support.capabilities);

    // setting the number of images in the swap chain.
    uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
    
    // must check if maxImageCount is set to 0 and clamp the image count by making sure it doesn't go over maxImageCount
    // if maxImageCount is set to 0, there would be an unlimited number of images until memory runs out
    if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount) {
      image_count = swap_chain_support.capabilities.maxImageCount;
    }

    // filling out the Vulkan object structure for the swap chain
    VkSwapchainCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface_;

    // the acquired information above is input here
    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1; // always set to 1 unless developing stereoscopic 3D application
    
    // this settings means will render directly to the image
    // VK_IMAGE_USAGE_TRANSFER_DST_BIT makes it so that you render to a separate image first for post processing
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; 
    
    // specifying how to handle swap chain images across multiple queue families
    QueueFamilyIndices indices = findQueueFamilies(physical_device_);
    
    // specifying which queue families are sharing images in VK_SHARING_MODE_CONCURRENT 
    uint32_t queue_family_indices[] = { (uint32_t)indices.graphics_family, (uint32_t)indices.present_family };

    // VK_SHARING_MODE_CONCURRENT allows images to be shared among multiple queue families
    if (indices.graphics_family != indices.present_family) {
      create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      create_info.queueFamilyIndexCount = 2;
      create_info.pQueueFamilyIndices = queue_family_indices;
    }

    // VK_SHARING_MODE_EXCLUSIVE makes it so an image has to be explicitly transferred from one queue family to another
    else {
      create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    create_info.preTransform = swap_chain_support.capabilities.currentTransform; // transforms applied to images in the swap chain
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // almost always want to ignore the alpha channel
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE; // obscured pixels won't be calculated

    if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create swap chain");
    }

    // retrieving handles just like any other retrieval of array of objects from Vulkan
    vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, nullptr);
    swap_chain_images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, swap_chain_images_.data());

    // storing format and extent to member variables for future use.
    swap_chain_image_format_ = surface_format.format;
    swap_chain_extent_ = extent;
  }

  //---------------------------------------------------------------------------
  // Purpose: creates a basic image view for every image in the swap chain
  //---------------------------------------------------------------------------
  void createImageViews() {
    swap_chain_image_views_.resize(swap_chain_images_.size());

    for (uint32_t i = 0; i < swap_chain_images_.size(); i++) {
      swap_chain_image_views_[i] = createImageView(swap_chain_images_[i], swap_chain_image_format_, VK_IMAGE_ASPECT_COLOR_BIT);
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: specify the number of colour and depth buffers, number of...
  // ...samples for each of the buffers and how to handle the contents...
  // ...wrapped in a renderpass object
  //---------------------------------------------------------------------------
  void createRenderPass() {
    VkAttachmentDescription colour_attachment = {};
    colour_attachment.format = swap_chain_image_format_; // should match the swap chain images
    colour_attachment.samples = VK_SAMPLE_COUNT_1_BIT;

    // decides what to do with the data in the attachment before rendering
    colour_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clearing the framebuffer before drawing a new frame
    colour_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // rendered contents will be stored in memory and can be read later

    // just like the above two parameters but for stencils
    colour_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; 
    colour_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    // setting the layout of pixels for the image
    colour_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // specifies the layout before the render pass begins
    colour_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // specifies the layout to automatically transition to when the render pass finishes

    VkAttachmentDescription depth_attachment = {};
    depth_attachment.format = findDepthFormat();
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colour_attachment_ref = {};
    colour_attachment_ref.attachment = 0; // attachment index
    colour_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // type of layout for the attachment

    VkAttachmentReference depth_attachment_ref = {};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // have to say explicitly that its a graphics subpass instead of a compute subpass
    subpass.colorAttachmentCount = 1; // the index for the fragment shader
    subpass.pColorAttachments = &colour_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // filling in the render pass struct with the attachments and subpass structs
    std::array<VkAttachmentDescription, 2> attachments = { colour_attachment, depth_attachment };
    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;

    if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create render pass!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: provide details about every descriptor binding used in the shaders
  //---------------------------------------------------------------------------
  void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding ubo_layout_binding = {};
    ubo_layout_binding.binding = 0;
    ubo_layout_binding.descriptorCount = 1;
    ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo_layout_binding.pImmutableSamplers = nullptr;
    ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding sampler_layout_binding = {};
    sampler_layout_binding.binding = 1;
    sampler_layout_binding.descriptorCount = 1;
    sampler_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sampler_layout_binding.pImmutableSamplers = nullptr;
    sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = { ubo_layout_binding, sampler_layout_binding };
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor set layout!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: Creating the pipeline and passing shaders through it
  //---------------------------------------------------------------------------
  void createGraphicsPipeline() {
    auto vert_shader_code = readFile("shaders/vert.spv");
    auto frag_shader_code = readFile("shaders/frag.spv");

    // the shader modules are only needed during the pipeline creation process so they are local
    VkShaderModule vert_shader_module = createShaderModule(vert_shader_code);
    VkShaderModule frag_shader_module = createShaderModule(frag_shader_code);

    VkPipelineShaderStageCreateInfo vert_shader_stage_info = {};
    vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_shader_stage_info.module = vert_shader_module; // implies that you can add multiple modules with different entry points
    vert_shader_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo frag_shader_stage_info = {};
    frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_shader_stage_info.module = frag_shader_module;
    frag_shader_stage_info.pName = "main";

    // store the shader stage info into an array for pipeline creation
    VkPipelineShaderStageCreateInfo shader_stages[] = { vert_shader_stage_info, frag_shader_stage_info };

    //----------------------------------
    /*   Fixed Function operations   */
    //----------------------------------
    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto binding_description = Vertex::getBindingDescription();
    auto attribute_descriptions = Vertex::getAttributeDescriptions();

    vertex_input_info.vertexBindingDescriptionCount = 1; // spacing between data and whether the data is per-vertex or per-instance
    vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size()); // type of the attributes passed to the vertex shader
    vertex_input_info.pVertexBindingDescriptions = &binding_description;
    vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.data();

    // input assembly describes the type of geometry from the vertices
    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // topology decides what kind of geometry will be drawn from the vertices
    input_assembly.primitiveRestartEnable = VK_FALSE; // allows you to use the element buffer

    // viewports describes the region of the framebuffer that the output will be rendered to
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swap_chain_extent_.width;
    viewport.height = (float)swap_chain_extent_.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    // scissor rectangles define the regions pixels will be stored
    // drawing the entire framebuffer so scissors cover it entirely
    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = swap_chain_extent_;

    // scissor and viewport values go in here
    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE; // this clamps fragments beyond the near and far planes instead of discarding them
    rasterizer.rasterizerDiscardEnable = VK_FALSE; // if VK_TRUE, the geometry never passes the rasterizer
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // determines how fragments are generated for geometry you could replace _FILL to _LINE or _POINT
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // determines the type of culling
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // determines the vertex order for faces to be considered front-facing
    rasterizer.depthBiasEnable = VK_FALSE; // alter the depth value

    // for anti-aliasing
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;

    // configuration per attached framebuffer
    VkPipelineColorBlendAttachmentState colour_blend_attachment = {};
    colour_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colour_blend_attachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colour_blending = {};
    colour_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colour_blending.logicOpEnable = VK_FALSE;
    colour_blending.logicOp = VK_LOGIC_OP_COPY;
    colour_blending.attachmentCount = 1;
    colour_blending.pAttachments = &colour_blend_attachment;
    colour_blending.blendConstants[0] = 0.0f;
    colour_blending.blendConstants[1] = 0.0f;
    colour_blending.blendConstants[2] = 0.0f;
    colour_blending.blendConstants[3] = 0.0f;

    // used to specify uniform values and push constants
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;

    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create pipeline layout!");
    }

    //-----------------------------------------
    // * End of Fixed Function Operations * //
    //-----------------------------------------

    // bring all the information together
    VkGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &colour_blending;
    pipeline_info.layout = pipeline_layout_;
    pipeline_info.renderPass = render_pass_;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create graphics pipeline!");
    }

    // cleaning up shader modules after pipeline creation
    vkDestroyShaderModule(device_, frag_shader_module, nullptr);
    vkDestroyShaderModule(device_, vert_shader_module, nullptr);
  }

  //---------------------------------------------------------------------------
  // Purpose: creating framebuffers for all the swap chain image views
  //---------------------------------------------------------------------------
  void createFramebuffers() {
    swap_chain_framebuffers_.resize(swap_chain_image_views_.size());

    // iterate through the image views and create framebuffers from them
    for (size_t i = 0; i < swap_chain_image_views_.size(); i++) {
      std::array<VkImageView, 2> attachments = {
        swap_chain_image_views_[i],
        depth_image_view_
      };

      VkFramebufferCreateInfo framebuffer_info = {};
      framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebuffer_info.renderPass = render_pass_;
      framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
      framebuffer_info.pAttachments = attachments.data();
      framebuffer_info.width = swap_chain_extent_.width;
      framebuffer_info.height = swap_chain_extent_.height;
      framebuffer_info.layers = 1;

      if (vkCreateFramebuffer(device_, &framebuffer_info, nullptr, &swap_chain_framebuffers_[i]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create framebuffer!");
      }
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: create a command pool that manages buffers
  //---------------------------------------------------------------------------
  void createCommandPool() {
    QueueFamilyIndices queue_family_indices = findQueueFamilies(physical_device_);

    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_indices.graphics_family;

    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create graphics command pool!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: set up resources for depth buffering
  //---------------------------------------------------------------------------
  void createDepthResources() {
    VkFormat depth_format = findDepthFormat();

    createImage(swap_chain_extent_.width, swap_chain_extent_.height, depth_format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image_, depth_image_memory_);
    depth_image_view_ = createImageView(depth_image_, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT);

    transitionImageLayout(depth_image_, depth_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  //---------------------------------------------------------------------------
  // Purpose: check for features
  //---------------------------------------------------------------------------
  VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physical_device_, format, &props);

      if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
        return format;
      }
      else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    throw std::runtime_error("Failed to find supported format!");
  }

  //---------------------------------------------------------------------------
  // Purpose: check if the device supports these depth buffer features
  //---------------------------------------------------------------------------
  VkFormat findDepthFormat() {
    return findSupportedFormat(
    { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
      VK_IMAGE_TILING_OPTIMAL,
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
  }

  //---------------------------------------------------------------------------
  // Purpose: check for stencil component
  //---------------------------------------------------------------------------
  bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
  }

  //---------------------------------------------------------------------------
  // Purpose: create an image for textures
  //---------------------------------------------------------------------------
  void createTextureImage() {
    // loading an image
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
    VkDeviceSize image_size = tex_width * tex_height * 4;

    if (!pixels) {
      throw std::runtime_error("Failed to load texture image!");
    }

    // copy image to temporary buffer
    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    createBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

    void* data;
    vkMapMemory(device_, staging_buffer_memory, 0, image_size, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(image_size));
    vkUnmapMemory(device_, staging_buffer_memory);

    // free the image
    stbi_image_free(pixels);

    createImage(tex_width, tex_height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture_image_, texture_image_memory_);

    transitionImageLayout(texture_image_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(staging_buffer, texture_image_, static_cast<uint32_t>(tex_width), static_cast<uint32_t>(tex_height));
    transitionImageLayout(texture_image_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_buffer_memory, nullptr);
  }

  //---------------------------------------------------------------------------
  // Purpose: create an image view for textures
  //---------------------------------------------------------------------------
  void createTextureImageView() {
    texture_image_view_ = createImageView(texture_image_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
  }

  //---------------------------------------------------------------------------
  // Purpose: set up the sampler for textures
  //---------------------------------------------------------------------------
  void createTextureSampler() {
    VkSamplerCreateInfo sampler_info = {};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = 16;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device_, &sampler_info, nullptr, &texture_sampler_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create texture sampler!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: create an image view
  //---------------------------------------------------------------------------
  VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags) {
    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D; // allows you to treat images as 1D, 2D, 3D textures or cube maps
    view_info.format = format;

    // these fields allow you to describe what the image's purpose is and which part is accessed
    // if a stereoscopic 3D application is being made, you can access different layers to make views for the left and right eye
    view_info.subresourceRange.aspectMask = aspect_flags;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView image_view;
    if (vkCreateImageView(device_, &view_info, nullptr, &image_view) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create texture image view!");
    }

    return image_view;
  }

  //---------------------------------------------------------------------------
  // Purpose: create an image object
  //---------------------------------------------------------------------------
  void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& image_memory) {
    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = tiling;
    image_info.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    image_info.usage = usage;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device_, &image_info, nullptr, &image) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create image!");
    }

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device_, image, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device_, &alloc_info, nullptr, &image_memory) != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate image memory!");
    }

    vkBindImageMemory(device_, image, image_memory, 0);
  }

  //---------------------------------------------------------------------------
  // Purpose: setting the image layout
  //---------------------------------------------------------------------------
  void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout) {
    VkCommandBuffer command_buffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;

    if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

      if (hasStencilComponent(format)) {
        barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      }
    }
    else {
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (old_layout == VK_IMAGE_LAYOUT_PREINITIALIZED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    else {
      throw std::invalid_argument("Unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &barrier
    );

    endSingleTimeCommands(command_buffer);
  }

  //---------------------------------------------------------------------------
  // Purpose: transfer data from buffer to image
  //---------------------------------------------------------------------------
  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer command_buffer = beginSingleTimeCommands();

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = {
      width,
      height,
      1
    };

    vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(command_buffer);
  }

  //---------------------------------------------------------------------------
  // Purpose: loading in the model from file and storing the vertices and indices
  //---------------------------------------------------------------------------
  void loadModel() {
    tinyobj::attrib_t attrib; // holds the positions, normals and texture coordinates
    std::vector<tinyobj::shape_t> shapes; // contains the objects and their faces
    std::vector<tinyobj::material_t> materials;
    std::string err; // errors and warnings when loading the file

    // loading the model into data structures
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, MODEL_PATH.c_str())) {
      throw std::runtime_error(err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

    // combine all of the faces into a single model
    for (const auto& shape : shapes) {
      for (const auto& index : shape.mesh.indices) { // assumes each vertex is unique
        Vertex vertex = {};

        vertex.pos = {
          attrib.vertices[3 * index.vertex_index + 0],
          attrib.vertices[3 * index.vertex_index + 1],
          attrib.vertices[3 * index.vertex_index + 2]
        };

        vertex.tex_coord = {
          attrib.texcoords[2 * index.texcoord_index + 0],
          1.0f - attrib.texcoords[2 * index.texcoord_index + 1] // inverting values because obj format assumes origin is bottom left corner instead of top left corner
        };

        vertex.colour = { 1.0f, 1.0f, 1.0f };

        if (uniqueVertices.count(vertex) == 0) {
          uniqueVertices[vertex] = static_cast<uint32_t>(vertices_.size());
          vertices_.push_back(vertex);
        }

        indices_.push_back(uniqueVertices[vertex]);
      }
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: create region of memory to store vertex buffer
  //---------------------------------------------------------------------------
  void createVertexBuffer() {
    VkDeviceSize buffer_size = sizeof(vertices_[0]) * vertices_.size();

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

    // filling the vertex buffer
    void* data;
    vkMapMemory(device_, staging_buffer_memory, 0, buffer_size, 0, &data);
    memcpy(data, vertices_.data(), (size_t)buffer_size);
    vkUnmapMemory(device_, staging_buffer_memory);

    createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer_, vertex_buffer_memory_);

    copyBuffer(staging_buffer, vertex_buffer_, buffer_size);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_buffer_memory, nullptr);
  }

  //---------------------------------------------------------------------------
  // Purpose: store indices for the geometry and send them to gpu buffer
  //---------------------------------------------------------------------------
  void createIndexBuffer() {
    VkDeviceSize buffer_size = sizeof(indices_[0]) * indices_.size();

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

    void* data;
    vkMapMemory(device_, staging_buffer_memory, 0, buffer_size, 0, &data);
    memcpy(data, indices_.data(), (size_t)buffer_size);
    vkUnmapMemory(device_, staging_buffer_memory);

    createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer_, index_buffer_memory_);

    copyBuffer(staging_buffer, index_buffer_, buffer_size);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_buffer_memory, nullptr);
  }

  //---------------------------------------------------------------------------
  // Purpose: buffer for the ubo data
  //---------------------------------------------------------------------------
  void createUniformBuffer() {
    VkDeviceSize buffer_size = sizeof(UniformBufferObject);
    createBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniform_buffer_, uniform_buffer_memory_);
  }

  //---------------------------------------------------------------------------
  // Purpose: buffer for the ubo data
  //---------------------------------------------------------------------------
  void createPerEyeUniformBuffer() {
    for (uint32_t eye = 0; eye < 2; eye++) {
      VkBufferCreateInfo buffer_create_info = {};
      buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      buffer_create_info.size = sizeof(glm::mat4);
      vkCreateBuffer(device_, &buffer_create_info, nullptr, &scene_uniform_buffer_[eye]);

      VkMemoryRequirements memory_requirements = {};
      vkGetBufferMemoryRequirements(device_, scene_uniform_buffer_[eye], &memory_requirements);
      
      VkMemoryAllocateInfo alloc_info = {};
      findMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
      alloc_info.allocationSize = memory_requirements.size;

      vkAllocateMemory(device_, &alloc_info, nullptr, &scene_uniform_buffer_memory_[eye]);
      vkBindBufferMemory(device_, scene_uniform_buffer_[eye], scene_uniform_buffer_memory_[eye], 0);

      void *data;

      vkMapMemory(device_, scene_uniform_buffer_memory_[eye], 0, VK_WHOLE_SIZE, 0, &data);
    }
  }

  void setCameras() {

  }

  //---------------------------------------------------------------------------
  // Purpose: create a pool for the descriptor sets
  //---------------------------------------------------------------------------
  void createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = 1;

    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor pool!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: allocating descriptor sets
  //---------------------------------------------------------------------------
  void createDescriptorSet() {
    VkDescriptorSetLayout layouts[] = { descriptor_set_layout_ };
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate descriptor set!");
    }

    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = uniform_buffer_;
    buffer_info.offset = 0;
    buffer_info.range = sizeof(UniformBufferObject);

    VkDescriptorImageInfo image_info = {};
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView = texture_image_view_;
    image_info.sampler = texture_sampler_;

    std::array<VkWriteDescriptorSet, 2> descriptor_writes = {};

    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = descriptor_set_;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &buffer_info;

    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].dstSet = descriptor_set_;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].pImageInfo = &image_info;

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
  }

  //---------------------------------------------------------------------------
  // Purpose: helper function to create buffers
  //---------------------------------------------------------------------------
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
    // create buffer
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size; // buffer size in bytes
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // just like the swap chain this determines who shares the data

    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create buffer!");
    }

    // assign memory to buffer
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate buffer memory!");
    }

    // associate memory with the buffer if allocation was successful
    vkBindBufferMemory(device_, buffer, buffer_memory, 0);
  }

  //---------------------------------------------------------------------------
  // Purpose: creating a temporary command buffer for memory operations
  //---------------------------------------------------------------------------
  VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool_;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(command_buffer, &begin_info);

    return command_buffer;
  }

  //---------------------------------------------------------------------------
  // Purpose: clearing temporary command buffers after memory operations
  //---------------------------------------------------------------------------
  void endSingleTimeCommands(VkCommandBuffer command_buffer) {
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue_);

    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
  }

  //---------------------------------------------------------------------------
  // Purpose: transferring data from staging buffer to destination buffer
  //---------------------------------------------------------------------------
  void copyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
    VkCommandBuffer command_buffer = beginSingleTimeCommands();

    VkBufferCopy copy_region = {};
    copy_region.size = size;
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

    endSingleTimeCommands(command_buffer);
  }

  //---------------------------------------------------------------------------
  // Purpose: combine memory requirements of the buffer and our own...
  // ...application requirements to find the right type of memory
  //---------------------------------------------------------------------------
  uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
      if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
  }

  //---------------------------------------------------------------------------
  // Purpose: allocates and records commands for each swap chain image
  //---------------------------------------------------------------------------
  void createCommandBuffers() {
    command_buffers_.resize(swap_chain_framebuffers_.size());

    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // determines if it's a primary buffer or a secondary buffer
    alloc_info.commandBufferCount = (uint32_t)command_buffers_.size();

    if (vkAllocateCommandBuffers(device_, &alloc_info, command_buffers_.data()) != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate command buffers!");
    }

    for (size_t i = 0; i < command_buffers_.size(); i++) {
      VkCommandBufferBeginInfo begin_info = {};
      begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; // specifies how we're going to use the command buffer

      vkBeginCommandBuffer(command_buffers_[i], &begin_info); // a call to this function implicitly resets the command buffer

      VkRenderPassBeginInfo render_pass_info = {};
      render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      render_pass_info.renderPass = render_pass_;
      render_pass_info.framebuffer = swap_chain_framebuffers_[i];

      // defining the size of the render area
      render_pass_info.renderArea.offset = { 0, 0 };
      render_pass_info.renderArea.extent = swap_chain_extent_;

      std::array<VkClearValue, 2> clear_values = {};
      clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
      clear_values[1].depthStencil = { 1.0f, 0 };

      render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
      render_pass_info.pClearValues = clear_values.data();

      vkCmdBeginRenderPass(command_buffers_[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

      // binding the graphics pipeline
      vkCmdBindPipeline(command_buffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

      // binding the vertex buffer during rendering
      VkBuffer vertex_buffers[] = { vertex_buffer_ };
      VkDeviceSize offsets[] = { 0 };
      vkCmdBindVertexBuffers(command_buffers_[i], 0, 1, vertex_buffers, offsets);

      vkCmdBindIndexBuffer(command_buffers_[i], index_buffer_, 0, VK_INDEX_TYPE_UINT32);
      vkCmdBindDescriptorSets(command_buffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

      // draw command
      vkCmdDrawIndexed(command_buffers_[i], static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);

      vkCmdEndRenderPass(command_buffers_[i]); // finishing the render pass

      if (vkEndCommandBuffer(command_buffers_[i]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
      }
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: creating semaphores to synchronise the rendering process
  //---------------------------------------------------------------------------
  void createSemaphores() {
    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    if (vkCreateSemaphore(device_, &semaphore_info, nullptr, &image_available_semaphore_) != VK_SUCCESS ||
      vkCreateSemaphore(device_, &semaphore_info, nullptr, &render_finished_semaphore_) != VK_SUCCESS) {

      throw std::runtime_error("Failed to create semaphores!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: generate a new transformation every frame
  //---------------------------------------------------------------------------
  void updateUniformBuffer() {
    static auto start_time = std::chrono::high_resolution_clock::now();

    auto current_time = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() / 1000.0f;

    UniformBufferObject ubo = {};
    ubo.model = glm::rotate(glm::mat4(), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), swap_chain_extent_.width / (float)swap_chain_extent_.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1; // flip the scaling factor of the Y axis in the projection matrix to take into account GLM's OpenGL layout

    // copy data to uniform buffer objects. push constants are an alternative to this and more efficient.
    void* data;
    vkMapMemory(device_, uniform_buffer_memory_, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device_, uniform_buffer_memory_);
  }

  //---------------------------------------------------------------------------
  // Purpose: acquire an image from the swap chain,...
  // ...execute the command buffer with that image as attachment in the framebuffer,...
  // ...return the image to the swapchain for presentation
  //---------------------------------------------------------------------------
  void drawFrame() {
    // acquire the image from the swap chain
    uint32_t image_index;
    VkResult result = vkAcquireNextImageKHR(device_, swap_chain_, std::numeric_limits<uint64_t>::max(), image_available_semaphore_, VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapChain();
      return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("Failed to acquire swap chain image!");
    }

    // execute the command buffer
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore wait_semaphores[] = { image_available_semaphore_ }; // setting which semaphore to wait for
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // which stage of the pipeline to wait
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffers_[image_index];

    VkSemaphore signal_semaphores[] = { render_finished_semaphore_ };
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    if (vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
      throw std::runtime_error("Failed to submit draw command buffer!");
    }

    // sending the result back to the swap chain
    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;

    VkSwapchainKHR swap_chains[] = { swap_chain_ };
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swap_chains;
    present_info.pImageIndices = &image_index;

    result = vkQueuePresentKHR(present_queue_, &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to present swap chain image!");
    }

    vkQueueWaitIdle(present_queue_);
  }

  //---------------------------------------------------------------------------
  // Purpose: create a VkShaderModule from shader code
  //---------------------------------------------------------------------------
  VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();

    // bytecode pointer is in uint32_t so a reinterpret_cast is needed
    // std::vector resolves alignment issues
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data()); 

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create shader module!");
    }

    return shader_module;
  }

  //---------------------------------------------------------------------------
  // Purpose: returns surface format values for the swap chain
  //---------------------------------------------------------------------------
  VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats) {
    // best case scenario is that the surface has no preferred format
    if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED) {
      return{ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    }

    // iterate through the list to see if a preferred combination is available
    for (const auto& available_format : available_formats) {
      if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return available_format;
      }
    }

    // if no preferred combination is available then select the first format specified
    return available_formats[0];
  }

  //---------------------------------------------------------------------------
  // Purpose: returns the best available presentation mode for the swap chain
  //---------------------------------------------------------------------------
  VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> available_present_modes) {
    VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto& available_present_mode : available_present_modes) {
      // looking for triple buffering if it's available
      if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return available_present_mode;
      }

      // some drivers don't properly support VK_PRESENT_MODE_FIFO_KHR
      // prefer VK_PRESENT_MODE_IMMEDIATE_KHR instead if above is not available
      else if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        best_mode = available_present_mode;
      }
    }

    return best_mode;
  }

  //---------------------------------------------------------------------------
  // Purpose: Setting the swap extent (resolution of the swap chain images)
  //---------------------------------------------------------------------------
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    else {
      int width, height;
      glfwGetWindowSize(companion_window_, &width, &height);

      VkExtent2D actual_extent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
      };

      actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
      actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

      return actual_extent;
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: populates SwapChainSupportDetails struct
  //---------------------------------------------------------------------------
  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, nullptr);

    if (format_count != 0) {
      details.formats.resize(format_count);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, details.formats.data());
    }

    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, nullptr);

    if (present_mode_count != 0) {
      details.present_modes.resize(present_mode_count);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, details.present_modes.data());
    }

    return details;
  }

  //---------------------------------------------------------------------------
  // Purpose: Checking physical device contains proper queue and extension...
  // ...support
  //---------------------------------------------------------------------------
  bool isDeviceSuitable(VkPhysicalDevice device) {
    // checks if device can process commands we want to use
    QueueFamilyIndices indices = findQueueFamilies(device);

    // important to only query for swap chain support after verifying the...
    // ...extension is available
    bool extensions_supported = checkDeviceExtensionSupport(device);

    // checking if the swap chain support is sufficient
    bool swap_chain_adequate = false;
    if (extensions_supported) {
      SwapChainSupportDetails swap_chain_support = querySwapChainSupport(device);
      swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.present_modes.empty();
    }

    VkPhysicalDeviceFeatures supported_features;
    vkGetPhysicalDeviceFeatures(device, &supported_features);

    // swap_chain_adequate not being used here??
    return indices.isComplete() && extensions_supported && supported_features.samplerAnisotropy;
  }

  //---------------------------------------------------------------------------
  // Purpose: check extension support on the device
  //---------------------------------------------------------------------------
  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    // enumerate the extensions
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

    // the set represents unconfirmed required extensions
    std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

    // check if all required extensions are available
    for (const auto& extension : available_extensions) {
      required_extensions.erase(extension.extensionName);
    }

    return required_extensions.empty();
  }

  //---------------------------------------------------------------------------
  // Purpose: function returns the indices of the queue families that... 
  // ...satisfy certain desired properties
  //---------------------------------------------------------------------------
  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    // getting the count
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    // getting the list of queue families
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    // check if the queue family has the required operations
    int i = 0;
    for (const auto& queue_family : queue_families) {
      if (queue_family.queueCount > 0 && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphics_family = i;
      }

      // checking queue families for presentation support
      VkBool32 present_support = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_support);

      if (queue_family.queueCount > 0 && present_support) {
        indices.present_family = i;
      }

      if (indices.isComplete()) {
        break;
      }

      i++;
    }

    return indices;
  }

  //---------------------------------------------------------------------------
  // Purpose: Find the required extensions to have Vulkan working
  //---------------------------------------------------------------------------
  std::vector<const char*> getRequiredExtensions() {
    std::vector<const char*> extensions;

    unsigned int glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    for (unsigned int i = 0; i < glfw_extension_count; i++) {
      extensions.push_back(glfw_extensions[i]);
    }

    //---------//
    // VR code //
    //---------//
    if (!vr::VRCompositor()) {
      throw std::runtime_error("Couldn't get the VR compositor for extensions");
    }

    uint32_t nBufferSize = vr::VRCompositor()->GetVulkanInstanceExtensionsRequired(nullptr, 0);
    if (nBufferSize > 0) {
      // Allocate memory for the space separated list and query for it
      char *pExtensionStr = new char[nBufferSize];
      pExtensionStr[0] = 0;
      vr::VRCompositor()->GetVulkanInstanceExtensionsRequired(pExtensionStr, nBufferSize);

      // Break up the space separated list into entries on the CUtlStringList
      std::string curExtStr;
      uint32_t nIndex = 0;
      while (pExtensionStr[nIndex] != 0 && (nIndex < nBufferSize)) {
        if (pExtensionStr[nIndex] == ' ') {
          const char* temp = strdup(curExtStr.c_str());
          extensions.push_back(temp);
          curExtStr.clear();
        }
        else {
          curExtStr += pExtensionStr[nIndex];
        }
        nIndex++;
      }
      if (curExtStr.size() > 0) {
        const char* temp = strdup(curExtStr.c_str());
        extensions.push_back(temp);
      }

      delete[] pExtensionStr;
    }
    //----------------//
    // End of VR code //
    //----------------//

    // added to receive messages from the validation layers
    if (enable_validation_layers) {
      // VK_EXT_DEBUG_REPORT_EXTENSION_NAME is a macro that avoids typos
      extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions;
  }

  //---------------------------------------------------------------------------
  // Purpose: check if validation layer is supported
  //---------------------------------------------------------------------------
  bool checkValidationLayerSupport() {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    // list all available validation layer extensions
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    // check if all of the layers in validation_layers exist in available_layers
    for (const char* layer_name : validation_layers) {
      bool layer_found = false;

      for (const auto& layer_properties : available_layers) {
        if (strcmp(layer_name, layer_properties.layerName) == 0) {
          layer_found = true;
          break;
        }
      }

      if (!layer_found) {
        return false;
      }
    }

    return true;
  }

  //---------------------------------------------------------------------------
  // Purpose: helper function to read in shader files
  //---------------------------------------------------------------------------
  static std::vector<char> readFile(const std::string& filename) {
    // std::ios::ate starts reading at the end of the file
    // std::ios::binary reads the file as binary so it would avoid text transformations
    std::ifstream file(filename, std::ios::ate | std::ios::binary); 

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file!");
    }

    // determine the filesize by read position
    size_t file_size = (size_t)file.tellg();
    std::vector<char> buffer(file_size);

    // seek to the beginning of the file and read all the bytes at once
    file.seekg(0);
    file.read(buffer.data(), file_size);

    // close the file and return the bytes
    file.close();

    return buffer;
  }

  //---------------------------------------------------------------------------
  // VKAPI_ATTR and VKAPI_CALL ensure the function has the right signature...
  // ...for Vulkan to call it
  // *first param: type of message
  // *second param: specifies the type of object that is the subject of...
  // ...the message
  // *eighth param: contains the message itself
  // *ninth param: you can pass your own data to the callback
  // This function is used to test validation layers themselves
  //---------------------------------------------------------------------------
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags,
    VkDebugReportObjectTypeEXT obj_type, uint64_t obj, size_t location, int32_t code,
    const char* layer_prefix, const char* msg, void* user_data)
  {
    std::cerr << "Validation layer: " << msg << std::endl;

    return VK_FALSE;
  }
};

//---------------------------------------------------------------------------
// Purpose: 
//---------------------------------------------------------------------------
int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  }
  catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}