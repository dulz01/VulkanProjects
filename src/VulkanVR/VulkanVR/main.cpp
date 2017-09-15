// Vulkan header integrated with GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION // includes the function bodies from stb_image.h
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION // includes the function bodies from tiny_obj_loader.h 
#include <tiny_obj_loader.h>

#include <openvr.h>

// Reporting and propagating errors
#include <stdexcept>
#include <iostream>

#include <deque>
#include <cstring> // for strcmp function
#include <algorithm> // for max and min functions
#include <fstream> // for file reading
#include <chrono> // timekeeping
#include <vector>
#include <array>
#include <set>
#include <unordered_map>

#include "Matrices.h"

// path to model and texture
const std::string MODEL_PATH = "models/chalet.obj";
const std::string TEXTURE_PATH = "textures/chalet.jpg";

//-----------------------------------------------------------------------------
// Purpose: shows which validation layers are required
//-----------------------------------------------------------------------------
// "VK_LAYER_LUNARG_standard_validation"
//"VK_LAYER_GOOGLE_threading",
//"VK_LAYER_LUNARG_parameter_validation",
//"VK_LAYER_LUNARG_object_tracker",
//"VK_LAYER_LUNARG_image",
//"VK_LAYER_LUNARG_core_validation",
//"VK_LAYER_LUNARG_swapchain"
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
  const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
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
  Vector3 pos;
  Vector3 colour;
  Vector2 tex_coord;

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
// Purpose: data we want the vertex shader to have
//-----------------------------------------------------------------------------
struct UniformBufferObject {
  Matrix4 model;
  Matrix4 view;
  Matrix4 proj;
};

//-----------------------------------------------------------------------------
// Purpose: The number of pipeline state objects
//-----------------------------------------------------------------------------
enum PipelineStateObjectEnum_t {
  PSO_SCENE = 0,
  PSO_COMPANION = 1,
  PSO_COUNT
};

//-----------------------------------------------------------------------------
// Purpose: Descriptor sets for rendering
//-----------------------------------------------------------------------------
enum DescriptorSetIndex_t {
  DESCRIPTOR_SET_LEFT_EYE_SCENE = 0,
  DESCRIPTOR_SET_RIGHT_EYE_SCENE,
  DESCRIPTOR_SET_COMPANION_LEFT_TEXTURE,
  DESCRIPTOR_SET_COMPANION_RIGHT_TEXTURE,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL0,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL1,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL2,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL3,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL4,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL5,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL6,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL7,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL8,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL9,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL10,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL11,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL12,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL13,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL14,
  DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL15,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL0,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL1,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL2,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL3,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL4,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL5,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL6,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL7,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL8,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL9,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL10,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL11,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL12,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL13,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL14,
  DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL15,
  NUM_DESCRIPTOR_SETS
};

//-----------------------------------------------------------------------------
// Purpose: main application
//-----------------------------------------------------------------------------
class VulkanVRApplication {
public:
  VulkanVRApplication()
    : companion_window_(NULL)
    , companion_window_width_(640)
    , companion_window_height_(320)
    , hmd_(NULL)
    , valid_pose_count_(0)
    , instance_(VK_NULL_HANDLE)
    , device_(VK_NULL_HANDLE)
    , physical_device_(VK_NULL_HANDLE)
    , queue_(VK_NULL_HANDLE)
    , surface_(VK_NULL_HANDLE)
    , swapchain_(VK_NULL_HANDLE)
    , command_pool_(VK_NULL_HANDLE)
    , descriptor_pool_(VK_NULL_HANDLE)
    , str_pose_classes_("")
    , frame_index_(0)
    , current_swapchain_image_(0)
    , scene_vertex_buffer_(VK_NULL_HANDLE)
    , scene_vertex_buffer_memory_(VK_NULL_HANDLE)
    , scene_image_(VK_NULL_HANDLE)
    , scene_image_memory_(VK_NULL_HANDLE)
    , scene_image_view_(VK_NULL_HANDLE)
    , scene_sampler_(VK_NULL_HANDLE)
    , descriptor_set_layout_(VK_NULL_HANDLE)
    , pipeline_layout_(VK_NULL_HANDLE)
    , pipeline_cache_(VK_NULL_HANDLE)
    , companion_window_vertex_buffer_(VK_NULL_HANDLE)
    , companion_window_vertex_buffer_memory_(VK_NULL_HANDLE)
    , companion_window_index_buffer_(VK_NULL_HANDLE)
    , companion_window_index_buffer_memory_(VK_NULL_HANDLE) {
    memset(&left_eye_desc_, 0, sizeof(left_eye_desc_));
    memset(&right_eye_desc_, 0, sizeof(right_eye_desc_));
    memset(&shader_modules_[0], 0, sizeof(shader_modules_));
    memset(&pipelines_[0], 0, sizeof(pipelines_));
    memset(scene_constant_buffer_data_, 0, sizeof(scene_constant_buffer_data_));
    memset(descriptor_sets_, 0, sizeof(descriptor_sets_));
  }

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
  VkBuffer companion_window_vertex_buffer_;
  VkDeviceMemory companion_window_vertex_buffer_memory_;
  VkBuffer companion_window_index_buffer_;
  VkDeviceMemory companion_window_index_buffer_memory_;
  unsigned int companion_window_index_size_;
  uint32_t companion_window_width_;
  uint32_t companion_window_height_;

  VkInstance instance_;
  VkDebugReportCallbackEXT callback_;
  VkSurfaceKHR surface_;

  VkPhysicalDevice physical_device_;
  VkPhysicalDeviceProperties physical_device_properties;
  VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
  VkPhysicalDeviceFeatures physical_device_features;
  uint32_t queue_family_index_;

  VkDevice device_;

  VkQueue queue_;
  VkQueue present_queue_; // DELETE

  VkSwapchainKHR swapchain_;
  uint32_t frame_index_;
  VkExtent2D swapchain_extent_;
  VkFormat swapchain_image_format_;
  uint32_t current_swapchain_image_;
  std::vector<VkImage> swapchain_images_; // implicitly created and destroyed
  std::vector<VkImageView> swapchain_image_views_;
  std::vector<VkFramebuffer> swapchain_framebuffers_;
  std::vector<VkSemaphore> swapchain_semaphores_;
  VkRenderPass swapchain_render_pass_;

  VkDescriptorSetLayout descriptor_set_layout_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline pipelines_[PSO_COUNT];
  VkPipelineCache pipeline_cache_;

  VkCommandPool command_pool_;

  std::vector<float> vertices_;
  unsigned int vert_count_;



  VkDescriptorPool descriptor_pool_;
  VkDescriptorSet descriptor_sets_[NUM_DESCRIPTOR_SETS];

  //------------------//
  // variables for VR //
  //------------------//
  vr::IVRSystem *hmd_;
  vr::IVRRenderModels *p_render_models_;
  vr::TrackedDevicePose_t tracked_device_pose_[vr::k_unMaxTrackedDeviceCount];
  int valid_pose_count_;
  std::string str_pose_classes_;
  char dev_class_char[vr::k_unMaxTrackedDeviceCount];

  Matrix4 mat4_device_pose_[vr::k_unMaxTrackedDeviceCount];
  Matrix4 mat4_hmd_pose_;
  Matrix4 mat4_proj_left_;
  Matrix4 mat4_proj_right_;
  Matrix4 mat4_eye_pos_left_;
  Matrix4 mat4_eye_pos_right_;

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
    VkFramebuffer framebuffer;
  };
  FramebufferDesc left_eye_desc_;
  FramebufferDesc right_eye_desc_;

  float near_clip_;
  float far_clip_;

  uint32_t render_width_;
  uint32_t render_height_;

  struct VulkanCommandBuffer_t {
    VkCommandBuffer command_buffer;
    VkFence fence;
  };
  std::deque<VulkanCommandBuffer_t> command_buffers_;
  VulkanCommandBuffer_t current_command_buffer_;

  VkBuffer scene_vertex_buffer_;
  VkDeviceMemory scene_vertex_buffer_memory_;
  VkBuffer scene_constant_buffer_[2];
  VkBuffer scene_constant_buffer_memory_[2];
  void* scene_constant_buffer_data_[2];
  VkImage scene_image_;
  VkDeviceMemory scene_image_memory_;
  VkImageView scene_image_view_;
  VkSampler scene_sampler_;

  VkShaderModule shader_modules_[PSO_COUNT * 2];

  struct VertexDataWindow {
    Vector2 position;
    Vector2 tex_coord;

    VertexDataWindow(const Vector2 &pos, const Vector2 tex) : position(pos), tex_coord(tex) {}
  };


  struct VertexDataScene {
    Vector3 position;
    Vector2 tex_coord;
  };

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
    vr::EVRInitError e_error = vr::VRInitError_None;
    hmd_ = vr::VR_Init(&e_error, vr::VRApplication_Scene);
    if (e_error != vr::VRInitError_None) {
      hmd_ = NULL;
      throw std::runtime_error("VR_Init Failed");
    }

    near_clip_ = 0.1f;
    far_clip_ = 30.0f;
    vert_count_ = 0;
    companion_window_index_size_ = 0;
    //----------------//
    // End of VR code //
    //----------------//

    // store a reference to the window when creating it
    companion_window_ = glfwCreateWindow(companion_window_width_, companion_window_height_, "VulkanVR", nullptr, nullptr);

    glfwSetWindowUserPointer(companion_window_, this);
  }

  //---------------------------------------------------------------------------
  // Purpose: calls all the functions required to initialise Vulkan
  //---------------------------------------------------------------------------
  void initVulkan() {
    InitVulkanInstance();
    InitVulkanDevice();
    InitVulkanSwapchain();

    createCommandPool();

    current_command_buffer_ = getCommandBuffer();
    VkCommandBufferBeginInfo command_buffer_begin_info = {};
    command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(current_command_buffer_.command_buffer, &command_buffer_begin_info);

    setupTexturemaps();
    loadModel();
    setupCameras();
    setupStereoRenderTargets();
    setupCompanionWindow();
    createGraphicsPipeline();
    createDescriptorSets();


    vkEndCommandBuffer(current_command_buffer_.command_buffer);
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &current_command_buffer_.command_buffer;
    vkQueueSubmit(queue_, 1, &submit_info, current_command_buffer_.fence);
    command_buffers_.push_front(current_command_buffer_);

    vkQueueWaitIdle(queue_);
  }

  //---------------------------------------------------------------------------
  // Purpose: 
  //---------------------------------------------------------------------------
  void mainLoop() {

    glfwSetInputMode(companion_window_, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    // keeping the application running until an error...
    // ... or the window is closed
    while (!glfwWindowShouldClose(companion_window_)) {
      glfwPollEvents();
      drawFrame();
    }

    vkDeviceWaitIdle(device_);
  }

  //---------------------------------------------------------------------------
  // Purpose: Destroying all the Vulkan objects explicitly created by us
  //---------------------------------------------------------------------------
  void cleanup() {
    if (device_ != VK_NULL_HANDLE) {
      vkDeviceWaitIdle(device_);
    }

    if (hmd_) {
      vr::VR_Shutdown();
      hmd_ = NULL;
    }

    if (device_ != VK_NULL_HANDLE) {
      for (std::deque<VulkanCommandBuffer_t>::iterator i = command_buffers_.begin(); i != command_buffers_.end(); i++) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &i->command_buffer);
        vkDestroyFence(device_, i->fence, nullptr);
      }
    }

    vkDestroyCommandPool(device_, command_pool_, nullptr);
    vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);

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
        vkDestroyFramebuffer(device_, pFramebufferDescs[i]->framebuffer, nullptr);
      }
    }

    vkDestroyImageView(device_, scene_image_view_, nullptr);
    vkDestroyImage(device_, scene_image_, nullptr);
    vkFreeMemory(device_, scene_image_memory_, nullptr);
    vkDestroySampler(device_, scene_sampler_, nullptr);
    vkDestroyBuffer(device_, scene_vertex_buffer_, nullptr);
    vkFreeMemory(device_, scene_vertex_buffer_memory_, nullptr);

    for (uint32_t eye = 0; eye < _countof(scene_constant_buffer_); eye++) {
      vkDestroyBuffer(device_, scene_constant_buffer_[eye], nullptr);
      vkFreeMemory(device_, scene_constant_buffer_memory_[eye], nullptr);
    }

    vkDestroyBuffer(device_, companion_window_vertex_buffer_, nullptr);
    vkFreeMemory(device_, companion_window_vertex_buffer_memory_, nullptr);
    vkDestroyBuffer(device_, companion_window_index_buffer_, nullptr);
    vkFreeMemory(device_, companion_window_index_buffer_memory_, nullptr);

    vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);

    for (uint32_t PSO = 0; PSO < PSO_COUNT; PSO++) {
      vkDestroyPipeline(device_, pipelines_[PSO], nullptr);
    }
    for (uint32_t shader = 0; shader < _countof(shader_modules_); shader++) {
      vkDestroyShaderModule(device_, shader_modules_[shader], nullptr);
    }
    vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);

    DestroyDebugReportCallbackEXT(instance_, callback_, nullptr);

    for (size_t swapchain_index = 0; swapchain_index < swapchain_framebuffers_.size(); swapchain_index++) {
      vkDestroyFramebuffer(device_, swapchain_framebuffers_[swapchain_index], nullptr);
      vkDestroyImageView(device_, swapchain_image_views_[swapchain_index], nullptr);
      vkDestroySemaphore(device_, swapchain_semaphores_[swapchain_index], nullptr);
    }
    vkDestroyRenderPass(device_, swapchain_render_pass_, nullptr);

    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);

    // clean up resources and terminating GLFW
    glfwDestroyWindow(companion_window_);
    glfwTerminate();
  }

  //---------------------------------------------------------------------------
  // Purpose: initializing Vulkan with an instance
  //---------------------------------------------------------------------------
  void InitVulkanInstance() {
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

    if (!enable_validation_layers) { return; }

    VkDebugReportCallbackCreateInfoEXT debug_create_info = {};
    debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;

    // allows you to filter what type of messages you would like to receive
    debug_create_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;

    // specifies the pointer to the callback function
    debug_create_info.pfnCallback = debugCallback;

    if (CreateDebugReportCallbackEXT(instance_, &debug_create_info, nullptr, &callback_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to set up debug callback!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: find a graphics card and check if it supports any Vulkan features
  //---------------------------------------------------------------------------
  void InitVulkanDevice() {
    if (glfwCreateWindowSurface(instance_, companion_window_, nullptr, &surface_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create window surface!");
    }

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

    vkGetPhysicalDeviceProperties(physical_device_, &physical_device_properties);
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &physical_device_memory_properties);
    vkGetPhysicalDeviceFeatures(physical_device_, &physical_device_features);

    // Logical device creation
    //QueueFamilyIndices indices = findQueueFamilies(physical_device_);
    QueueFamilyIndices indices;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    // check if the queue family has the required operations
    uint32_t i = 0;
    for (i = 0; i < queue_family_count; i++) {
      if (queue_families[i].queueCount > 0 && queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        break;
      }
    }
    if (i >= queue_family_count) {
      throw std::runtime_error("No graphics queue found");
    }
    queue_family_index_ = i;

    VkDeviceQueueCreateInfo device_queue_create_info = {};
    device_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    device_queue_create_info.queueFamilyIndex = queue_family_index_;
    device_queue_create_info.queueCount = 1;
    float queue_priority = 1.0f;
    device_queue_create_info.pQueuePriorities = &queue_priority;

    // specifying what set of device features we'll be using
    VkPhysicalDeviceFeatures device_features = {};
    vkGetPhysicalDeviceFeatures(physical_device_, &device_features);

    // creating the logical device
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &device_queue_create_info;
    create_info.pEnabledFeatures = &device_features;
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
    vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);
  }

  //---------------------------------------------------------------------------
  // Purpose: create the swap chain
  //---------------------------------------------------------------------------
  void InitVulkanSwapchain() {
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
    VkSwapchainCreateInfoKHR swapchain_create_info = {};
    swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_create_info.surface = surface_;
    swapchain_create_info.minImageCount = image_count;
    swapchain_create_info.imageFormat = surface_format.format;
    swapchain_create_info.imageColorSpace = surface_format.colorSpace;
    swapchain_create_info.imageExtent = extent;
    swapchain_create_info.imageArrayLayers = 1; // always set to 1 unless developing stereoscopic 3D application
    swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // VK_IMAGE_USAGE_TRANSFER_DST_BIT makes it so that you render to a separate image first for post processing
    swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_create_info.queueFamilyIndexCount = 0;
    swapchain_create_info.pQueueFamilyIndices = NULL;
    swapchain_create_info.preTransform = swap_chain_support.capabilities.currentTransform; // transforms applied to images in the swap chain
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // almost always want to ignore the alpha channel
    swapchain_create_info.presentMode = present_mode;
    swapchain_create_info.clipped = VK_TRUE; // obscured pixels won't be calculated

    if (vkCreateSwapchainKHR(device_, &swapchain_create_info, nullptr, &swapchain_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create swap chain");
    }

    // retrieving handles just like any other retrieval of array of objects from Vulkan
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    swapchain_images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());

    // storing format and extent to member variables for future use.
    swapchain_image_format_ = surface_format.format;
    swapchain_extent_ = extent;

    // Create renderpass
    uint32_t total_attachments = 1;
    VkAttachmentReference attachment_ref = {};
    attachment_ref.attachment = 0; // attachment index
    attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // type of layout for the attachment

    VkAttachmentDescription attachment_desc = {};
    attachment_desc.format = swapchain_image_format_; // should match the swap chain images
    attachment_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clearing the framebuffer before drawing a new frame
    attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // rendered contents will be stored in memory and can be read later
    attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment_desc.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // specifies the layout before the render pass begins
    attachment_desc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // specifies the layout to automatically transition to when the render pass finishes
    attachment_desc.flags = 0;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // have to say explicitly that its a graphics subpass instead of a compute subpass
    subpass.flags = 0;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = NULL;
    subpass.colorAttachmentCount = 1; // the index for the fragment shader
    subpass.pColorAttachments = &attachment_ref;
    subpass.pResolveAttachments = NULL;
    subpass.pDepthStencilAttachment = NULL;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = NULL;

    // filling in the render pass struct with the attachments and subpass structs
    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &attachment_desc;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 0;
    render_pass_info.pDependencies = NULL;

    if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &swapchain_render_pass_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create render pass!");
    }

    // Create image views
    swapchain_image_views_.resize(swapchain_images_.size());
    swapchain_framebuffers_.resize(swapchain_images_.size());

    for (uint32_t i = 0; i < swapchain_images_.size(); i++) {
      VkImageViewCreateInfo view_info = {};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.flags = 0;
      view_info.image = swapchain_images_[i];
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D; // allows you to treat images as 1D, 2D, 3D textures or cube maps
      view_info.format = swapchain_image_format_;
      view_info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view_info.subresourceRange.baseMipLevel = 0;
      view_info.subresourceRange.levelCount = 1;
      view_info.subresourceRange.baseArrayLayer = 0;
      view_info.subresourceRange.layerCount = 1;

      if (vkCreateImageView(device_, &view_info, nullptr, &swapchain_image_views_[i]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture image view!");
      }

      VkImageView attachments[1] = { swapchain_image_views_[i] };
      VkFramebufferCreateInfo framebuffer_create_info = {};
      framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebuffer_create_info.renderPass = swapchain_render_pass_;
      framebuffer_create_info.attachmentCount = 1;
      framebuffer_create_info.pAttachments = &attachments[0];
      framebuffer_create_info.width = companion_window_width_;
      framebuffer_create_info.height = companion_window_height_;
      framebuffer_create_info.layers = 1;

      if (vkCreateFramebuffer(device_, &framebuffer_create_info, nullptr, &swapchain_framebuffers_[i]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create framebuffer!");
      }

      VkSemaphoreCreateInfo semaphore_create_info = {};
      semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      VkSemaphore semaphore = VK_NULL_HANDLE;
      vkCreateSemaphore(device_, &semaphore_create_info, nullptr, &semaphore);
      swapchain_semaphores_.push_back(semaphore);
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
    const char *shader_names[PSO_COUNT] =
    {
      "scene",
      "companion"
    };
    const char *stage_names[2] =
    {
      "vs",
      "ps"
    };

    // Load the SPIR-V into shader modules
    for (int32_t shader = 0; shader < PSO_COUNT; shader++) {
      for (int32_t stage = 0; stage <= 1; stage++) {
        char shader_file_name[1024];
        sprintf(shader_file_name, "shaders/%s_%s.spv", shader_names[shader], stage_names[stage]);
        std::string shader_path = shader_file_name;

        FILE *fp = fopen(shader_path.c_str(), "rb");
        if (fp == NULL) {
          throw std::runtime_error("Error opening SPIR-V file");
        }
        fseek(fp, 0, SEEK_END);
        size_t size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        char *buffer = new char[size];
        if (fread(buffer, 1, size, fp) != size) {
          throw std::runtime_error("Error opening SPIR-V file");
        }
        fclose(fp);

        // Create the shader module
        VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
        shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleCreateInfo.codeSize = size;
        shaderModuleCreateInfo.pCode = (const uint32_t *)buffer;
        if (vkCreateShaderModule(device_, &shaderModuleCreateInfo, nullptr, &shader_modules_[shader * 2 + stage]) != VK_SUCCESS) {
          throw std::runtime_error("Error creating shader module");
        }

        delete[] buffer;
      }
    }

    // Create a descriptor set layout/pipeline layout compatible with all of our shaders.  See bin/shaders/build_vulkan_shaders.bat for
    // how the HLSL is compiled with glslangValidator and binding numbers are generated
    VkDescriptorSetLayoutBinding layout_bindings[3] = {};
    layout_bindings[0].binding = 0;
    layout_bindings[0].descriptorCount = 1;
    layout_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layout_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    layout_bindings[1].binding = 1;
    layout_bindings[1].descriptorCount = 1;
    layout_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    layout_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    layout_bindings[2].binding = 2;
    layout_bindings[2].descriptorCount = 1;
    layout_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    layout_bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
    descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_set_layout_create_info.bindingCount = 3;
    descriptor_set_layout_create_info.pBindings = &layout_bindings[0];

    vkCreateDescriptorSetLayout(device_, &descriptor_set_layout_create_info, nullptr, &descriptor_set_layout_);

    VkPipelineLayoutCreateInfo pipeline_layout_create_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_create_info.pNext = NULL;
    pipeline_layout_create_info.setLayoutCount = 1;
    pipeline_layout_create_info.pSetLayouts = &descriptor_set_layout_;
    pipeline_layout_create_info.pushConstantRangeCount = 0;
    pipeline_layout_create_info.pPushConstantRanges = NULL;

    vkCreatePipelineLayout(device_, &pipeline_layout_create_info, nullptr, &pipeline_layout_);

    // Create pipeline cache
    VkPipelineCacheCreateInfo pipeline_cache_create_info = {};
    pipeline_cache_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkCreatePipelineCache(device_, &pipeline_cache_create_info, NULL, &pipeline_cache_);

    // Renderpass for each PSO that is compatible with what it will render to
    VkRenderPass render_passes[PSO_COUNT] = {
      left_eye_desc_.render_pass, // PSO_SCENE
      swapchain_render_pass_      // PSO_COMPANIOH
    };

    size_t strides[PSO_COUNT] =
    {
      sizeof(VertexDataScene), // PSO_SCENE
      sizeof(VertexDataWindow) // PSO_COMPANIOH
    };

    VkVertexInputAttributeDescription attribute_descriptions[PSO_COUNT * 2]{
      // PSO_SCENE
      { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,	0 },
      { 1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexDataScene, tex_coord) },

      // PSO_COMPANION
      { 0, 0, VK_FORMAT_R32G32_SFLOAT,	0 },
      { 1, 0, VK_FORMAT_R32G32_SFLOAT,	sizeof(float) * 2 },
    };

    // Create the PSOs
    for (uint32_t PSO = 0; PSO < PSO_COUNT; PSO++) {
      VkGraphicsPipelineCreateInfo pipeline_create_info = {};
      pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

      VkVertexInputBindingDescription binding_description;
      binding_description.binding = 0;
      binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      binding_description.stride = strides[PSO];

      VkPipelineVertexInputStateCreateInfo vertex_input_create_info = {};
      vertex_input_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      for (uint32_t attrib = 0; attrib < 2; attrib++) {
        if (attribute_descriptions[PSO * 2 + attrib].format != VK_FORMAT_UNDEFINED) {
          vertex_input_create_info.vertexAttributeDescriptionCount++;
        }
      }
      vertex_input_create_info.pVertexAttributeDescriptions = &attribute_descriptions[PSO * 2];
      vertex_input_create_info.vertexBindingDescriptionCount = 1;
      vertex_input_create_info.pVertexBindingDescriptions = &binding_description;

      VkPipelineDepthStencilStateCreateInfo depth_stencil_state = {};
      depth_stencil_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
      depth_stencil_state.depthTestEnable = (PSO != PSO_COMPANION) ? VK_TRUE : VK_FALSE;
      depth_stencil_state.depthWriteEnable = (PSO != PSO_COMPANION) ? VK_TRUE : VK_FALSE;
      depth_stencil_state.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      depth_stencil_state.depthBoundsTestEnable = VK_FALSE;
      depth_stencil_state.stencilTestEnable = VK_FALSE;
      depth_stencil_state.minDepthBounds = 0.0f;
      depth_stencil_state.maxDepthBounds = 0.0f;

      VkPipelineColorBlendStateCreateInfo colour_blend_state = {};
      colour_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      colour_blend_state.logicOpEnable = VK_FALSE;
      colour_blend_state.logicOp = VK_LOGIC_OP_COPY;

      VkPipelineColorBlendAttachmentState colour_blend_attachment = {};
      colour_blend_attachment.blendEnable = VK_FALSE;
      colour_blend_attachment.colorWriteMask = 0xf;

      colour_blend_state.attachmentCount = 1;
      colour_blend_state.pAttachments = &colour_blend_attachment;

      VkPipelineRasterizationStateCreateInfo rasterization_state = {};
      rasterization_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
      rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
      rasterization_state.cullMode = VK_CULL_MODE_BACK_BIT;
      rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      rasterization_state.lineWidth = 1.0f;

      VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
      input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      input_assembly.primitiveRestartEnable = VK_FALSE;

      VkPipelineMultisampleStateCreateInfo multisample_state = {};
      multisample_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
      multisample_state.rasterizationSamples = (PSO == PSO_COMPANION) ? VK_SAMPLE_COUNT_1_BIT : VK_SAMPLE_COUNT_4_BIT;
      multisample_state.minSampleShading = 0.0f;
      uint32_t nSampleMask = 0xFFFFFFFF;
      multisample_state.pSampleMask = &nSampleMask;

      VkPipelineViewportStateCreateInfo viewport = {};
      viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;

      VkPipelineShaderStageCreateInfo shader_stages[2] = {};
      shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
      shader_stages[0].module = shader_modules_[PSO * 2 + 0];
      shader_stages[0].pName = "VSMain";

      shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      shader_stages[1].module = shader_modules_[PSO * 2 + 1];
      shader_stages[1].pName = "PSMain";

      pipeline_create_info.layout = pipeline_layout_;

      // Set pipeline states
      pipeline_create_info.pVertexInputState = &vertex_input_create_info;
      pipeline_create_info.pInputAssemblyState = &input_assembly;
      pipeline_create_info.pViewportState = &viewport;
      pipeline_create_info.pRasterizationState = &rasterization_state;
      pipeline_create_info.pMultisampleState = &multisample_state;
      pipeline_create_info.pDepthStencilState = &depth_stencil_state;
      pipeline_create_info.pColorBlendState = &colour_blend_state;
      pipeline_create_info.stageCount = 2;
      pipeline_create_info.pStages = &shader_stages[0];
      pipeline_create_info.renderPass = render_passes[PSO];

      static VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
      };

      static VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
      dynamic_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dynamic_state_create_info.pNext = NULL;
      dynamic_state_create_info.dynamicStateCount = _countof(dynamic_states);
      dynamic_state_create_info.pDynamicStates = &dynamic_states[0];
      pipeline_create_info.pDynamicState = &dynamic_state_create_info;

      if (vkCreateGraphicsPipelines(device_, pipeline_cache_, 1, &pipeline_create_info, NULL, &pipelines_[PSO]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create the graphics pipeline");
      }
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: create a command pool that manages buffers
  //---------------------------------------------------------------------------
  void createCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_index_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create graphics command pool!");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: create an image for textures
  //---------------------------------------------------------------------------
  void setupTexturemaps() {
    // loading an image
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);

    if (!pixels) {
      throw std::runtime_error("Failed to load texture image!");
    }

    VkDeviceSize buffer_size = tex_width * tex_height * 4;

    std::vector<VkBufferImageCopy> buffer_image_copies;
    VkBufferImageCopy buffer_image_copy = {};
    buffer_image_copy.bufferOffset = 0;
    buffer_image_copy.bufferRowLength = 0;
    buffer_image_copy.bufferImageHeight = 0;
    buffer_image_copy.imageSubresource.baseArrayLayer = 0;
    buffer_image_copy.imageSubresource.layerCount = 1;
    buffer_image_copy.imageSubresource.mipLevel = 0;
    buffer_image_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    buffer_image_copy.imageOffset.x = 0;
    buffer_image_copy.imageOffset.y = 0;
    buffer_image_copy.imageOffset.z = 0;
    buffer_image_copy.imageExtent.width = tex_width;
    buffer_image_copy.imageExtent.height = tex_height;
    buffer_image_copy.imageExtent.depth = 1;
    buffer_image_copies.push_back(buffer_image_copy);

    // Create the image
    VkImageCreateInfo image_create_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.extent.width = tex_width;
    image_create_info.extent.height = tex_height;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_create_info.flags = 0;
    vkCreateImage(device_, &image_create_info, nullptr, &scene_image_);

    VkMemoryRequirements memory_requirements = {};
    vkGetImageMemoryRequirements(device_, scene_image_, &memory_requirements);

    VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    memoryAllocateInfo.allocationSize = memory_requirements.size;
    memoryAllocateInfo.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device_, &memoryAllocateInfo, nullptr, &scene_image_memory_);
    vkBindImageMemory(device_, scene_image_, scene_image_memory_, 0);

    // Create the image view
    VkImageViewCreateInfo image_view_create_info = {};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.flags = 0;
    image_view_create_info.image = scene_image_;
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = image_create_info.format;
    image_view_create_info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = 0;
    image_view_create_info.subresourceRange.levelCount = image_create_info.mipLevels;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount = 1;
    vkCreateImageView(device_, &image_view_create_info, nullptr, &scene_image_view_);

    // Create a staging buffer
    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    createAndFillBuffer(pixels, buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, staging_buffer, staging_buffer_memory);

    // Transition the image to TRANSFER_DST to receive image
    VkImageMemoryBarrier image_memory_barrier = {};
    image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.srcAccessMask = 0;
    image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_memory_barrier.image = scene_image_;
    image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_memory_barrier.subresourceRange.baseMipLevel = 0;
    image_memory_barrier.subresourceRange.levelCount = image_create_info.mipLevels;
    image_memory_barrier.subresourceRange.baseArrayLayer = 0;
    image_memory_barrier.subresourceRange.layerCount = 1;
    image_memory_barrier.srcQueueFamilyIndex = queue_family_index_;
    image_memory_barrier.dstQueueFamilyIndex = queue_family_index_;
    //vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);

    // Issue the copy to fill the image data
    vkCmdCopyBufferToImage(current_command_buffer_.command_buffer, staging_buffer, scene_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)buffer_image_copies.size(), &buffer_image_copies[0]);

    // Transition the image to SHADER_READ_OPTIMAL for reading
    image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);

    // Create the sampler
    VkSamplerCreateInfo sampler_create_info = {};
    sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_create_info.magFilter = VK_FILTER_LINEAR;
    sampler_create_info.minFilter = VK_FILTER_LINEAR;
    sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.anisotropyEnable = VK_TRUE;
    sampler_create_info.maxAnisotropy = 16.0f;
    sampler_create_info.minLod = 0.0f;
    sampler_create_info.maxLod = 0.0f;
    vkCreateSampler(device_, &sampler_create_info, nullptr, &scene_sampler_);

    stbi_image_free(pixels);
  }

  //---------------------------------------------------------------------------
  // Purpose: loading in the model from file and storing the vertices and indices
  //---------------------------------------------------------------------------
  void loadModel() {
    if (!hmd_) {
      return;
    }

    tinyobj::attrib_t attrib; // holds the positions, normals and texture coordinates
    std::vector<tinyobj::shape_t> shapes; // contains the objects and their faces
    std::vector<tinyobj::material_t> materials;
    std::string err; // errors and warnings when loading the file

    // loading the model into data structures
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, MODEL_PATH.c_str())) {
      throw std::runtime_error(err);
    }

    // combine all of the faces into a single model
    for (const auto& shape : shapes) {
      for (const auto& index : shape.mesh.indices) { // assumes each vertex is unique
        //Vertex vertex = {};
        vertices_.push_back(attrib.vertices[3 * index.vertex_index + 0]);
        vertices_.push_back(attrib.vertices[3 * index.vertex_index + 1]);
        vertices_.push_back(attrib.vertices[3 * index.vertex_index + 2]);
        vertices_.push_back(attrib.texcoords[2 * index.texcoord_index + 0]);
        vertices_.push_back(1.0f - attrib.texcoords[2 * index.texcoord_index + 1]); // inverting values because obj format assumes origin is bottom left corner instead of top left corner
        //vertex.pos = {
        //};

        //vertex.tex_coord = {
        //};

        //vertex.colour = { 1.0f, 1.0f, 1.0f };

        //vertices_.push_back(vertex);
      }
    }
    vert_count_ = vertices_.size();

    // Create the vertex buffer and fill with data
    createAndFillBuffer(&vertices_[0], vert_count_ * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, scene_vertex_buffer_, scene_vertex_buffer_memory_);

    // Create the constant buffer to hold the per-eye constant buffer data
    for (uint32_t eye = 0; eye < 2; eye++) {
      VkBufferCreateInfo buffer_create_info = {};
      buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      buffer_create_info.size = sizeof(Matrix4);
      buffer_create_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      vkCreateBuffer(device_, &buffer_create_info, nullptr, &scene_constant_buffer_[eye]);

      VkMemoryRequirements memory_requirements = {};
      vkGetBufferMemoryRequirements(device_, scene_constant_buffer_[eye], &memory_requirements);

      VkMemoryAllocateInfo alloc_info = {};
      alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      alloc_info.allocationSize = memory_requirements.size;
      alloc_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      vkAllocateMemory(device_, &alloc_info, nullptr, &scene_constant_buffer_memory_[eye]);
      vkBindBufferMemory(device_, scene_constant_buffer_[eye], scene_constant_buffer_memory_[eye], 0);

      // Keep map persistently
      vkMapMemory(device_, scene_constant_buffer_memory_[eye], 0, VK_WHOLE_SIZE, 0, &scene_constant_buffer_data_[eye]);
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: allocating descriptor sets
  //---------------------------------------------------------------------------
  void createDescriptorSets() {
    VkDescriptorPoolSize pool_sizes[3];
    pool_sizes[0].descriptorCount = NUM_DESCRIPTOR_SETS;
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = NUM_DESCRIPTOR_SETS;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[2].descriptorCount = NUM_DESCRIPTOR_SETS;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLER;

    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.flags = 0;
    descriptor_pool_create_info.maxSets = NUM_DESCRIPTOR_SETS;
    descriptor_pool_create_info.poolSizeCount = _countof(pool_sizes);
    descriptor_pool_create_info.pPoolSizes = &pool_sizes[0];

    vkCreateDescriptorPool(device_, &descriptor_pool_create_info, nullptr, &descriptor_pool_);

    for (int descriptor_set = 0; descriptor_set < NUM_DESCRIPTOR_SETS; descriptor_set++) {
      VkDescriptorSetAllocateInfo alloc_info = {};
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool = descriptor_pool_;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &descriptor_set_layout_;
      vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_sets_[descriptor_set]);
    }

    // Scene descriptor sets
    for (uint32_t eye = 0; eye < 2; eye++) {
      VkDescriptorBufferInfo buffer_info = {};
      buffer_info.buffer = scene_constant_buffer_[eye];
      buffer_info.offset = 0;
      buffer_info.range = VK_WHOLE_SIZE;

      VkDescriptorImageInfo image_info = {};
      image_info.imageView = scene_image_view_;
      image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      VkDescriptorImageInfo sampler_info = {};
      sampler_info.sampler = scene_sampler_;

      VkWriteDescriptorSet write_descriptor_sets[3] = {};
      write_descriptor_sets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_descriptor_sets[0].dstSet = descriptor_sets_[DESCRIPTOR_SET_LEFT_EYE_SCENE + eye];
      write_descriptor_sets[0].dstBinding = 0;
      write_descriptor_sets[0].descriptorCount = 1;
      write_descriptor_sets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      write_descriptor_sets[0].pBufferInfo = &buffer_info;
      write_descriptor_sets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_descriptor_sets[1].dstSet = descriptor_sets_[DESCRIPTOR_SET_LEFT_EYE_SCENE + eye];
      write_descriptor_sets[1].dstBinding = 1;
      write_descriptor_sets[1].descriptorCount = 1;
      write_descriptor_sets[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      write_descriptor_sets[1].pImageInfo = &image_info;
      write_descriptor_sets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_descriptor_sets[2].dstSet = descriptor_sets_[DESCRIPTOR_SET_LEFT_EYE_SCENE + eye];
      write_descriptor_sets[2].dstBinding = 2;
      write_descriptor_sets[2].descriptorCount = 1;
      write_descriptor_sets[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      write_descriptor_sets[2].pImageInfo = &sampler_info;

      vkUpdateDescriptorSets(device_, _countof(write_descriptor_sets), write_descriptor_sets, 0, nullptr);
    }

    // Companion window descriptor sets
    {
      VkDescriptorImageInfo imageInfo = {};
      imageInfo.imageView = left_eye_desc_.image_view;
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      VkWriteDescriptorSet write_descriptor_sets[1] = {};
      write_descriptor_sets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_descriptor_sets[0].dstSet = descriptor_sets_[DESCRIPTOR_SET_COMPANION_LEFT_TEXTURE];
      write_descriptor_sets[0].dstBinding = 1;
      write_descriptor_sets[0].descriptorCount = 1;
      write_descriptor_sets[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      write_descriptor_sets[0].pImageInfo = &imageInfo;
      vkUpdateDescriptorSets(device_, _countof(write_descriptor_sets), write_descriptor_sets, 0, nullptr);

      imageInfo.imageView = right_eye_desc_.image_view;
      write_descriptor_sets[0].dstSet = descriptor_sets_[DESCRIPTOR_SET_COMPANION_RIGHT_TEXTURE];
      vkUpdateDescriptorSets(device_, _countof(write_descriptor_sets), write_descriptor_sets, 0, nullptr);
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: acquire an image from the swap chain,...
  // ...execute the command buffer with that image as attachment in the framebuffer,...
  // ...return the image to the swapchain for presentation
  //---------------------------------------------------------------------------
  void drawFrame() {
    if (hmd_) {
      current_command_buffer_ = getCommandBuffer();

      // start the command buffer
      VkCommandBufferBeginInfo command_buffer_begin_info = {};
      command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(current_command_buffer_.command_buffer, &command_buffer_begin_info);

      renderStereoTargets();
      renderCompanionWindow();

      // end the command buffer
      vkEndCommandBuffer(current_command_buffer_.command_buffer);

      // submit the command buffer
      VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      VkSubmitInfo submit_info = {};
      submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.commandBufferCount = 1;
      submit_info.pCommandBuffers = &current_command_buffer_.command_buffer;
      submit_info.waitSemaphoreCount = 1;
      submit_info.pWaitSemaphores = &swapchain_semaphores_[frame_index_];
      submit_info.pWaitDstStageMask = &wait_dst_stage_mask;
      vkQueueSubmit(queue_, 1, &submit_info, current_command_buffer_.fence);

      // Add the command buffer back for later recycling
      command_buffers_.push_front(current_command_buffer_);

      // Submit to SteamVR
      vr::VRTextureBounds_t bounds;
      bounds.uMin = 0.0f;
      bounds.uMax = 1.0f;
      bounds.vMin = 0.0f;
      bounds.vMax = 1.0f;

      vr::VRVulkanTextureData_t vulkan_data;
      vulkan_data.m_nImage = (uint64_t)left_eye_desc_.image;
      vulkan_data.m_pDevice = (VkDevice_T *)device_;
      vulkan_data.m_pPhysicalDevice = (VkPhysicalDevice_T *)physical_device_;
      vulkan_data.m_pInstance = (VkInstance_T *)instance_;
      vulkan_data.m_pQueue = (VkQueue_T *)queue_;

      vulkan_data.m_nQueueFamilyIndex = queue_family_index_;

      vulkan_data.m_nWidth = render_width_;
      vulkan_data.m_nHeight = render_height_;
      vulkan_data.m_nFormat = VK_FORMAT_R8G8B8A8_SRGB;
      vulkan_data.m_nSampleCount = 4;

      vr::Texture_t texture = { &vulkan_data, vr::TextureType_Vulkan, vr::ColorSpace_Auto };
      vr::VRCompositor()->Submit(vr::Eye_Left, &texture, &bounds);

      vulkan_data.m_nImage = (uint64_t)right_eye_desc_.image;
      vr::VRCompositor()->Submit(vr::Eye_Right, &texture, &bounds);
    }
    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pNext = NULL;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_;
    present_info.pImageIndices = &current_swapchain_image_;
    vkQueuePresentKHR(queue_, &present_info);

    updateHMDMatrixPose();

    frame_index_ = (frame_index_ + 1) % swapchain_images_.size();
  }

  //---------------------------------------------------------------------------
  // Purpose: Render for VR headset
  //---------------------------------------------------------------------------
  void renderStereoTargets() {
    // set the viewport and scissor
    VkViewport viewport = { 0.0f, 0.0f, (float)render_width_, (float)render_height_, 0.0f, 1.0f };
    vkCmdSetViewport(current_command_buffer_.command_buffer, 0, 1, &viewport);
    VkRect2D scissor = { 0, 0, render_width_, render_height_ };
    vkCmdSetScissor(current_command_buffer_.command_buffer, 0, 1, &scissor);

    //----------//
    // Left Eye //
    //----------//
    //Transition eye image to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier image_memory_barrier = {};
    image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_memory_barrier.oldLayout = left_eye_desc_.image_layout;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    image_memory_barrier.image = left_eye_desc_.image;
    image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_memory_barrier.subresourceRange.baseMipLevel = 0;
    image_memory_barrier.subresourceRange.levelCount = 1;
    image_memory_barrier.subresourceRange.baseArrayLayer = 0;
    image_memory_barrier.subresourceRange.layerCount = 1;
    image_memory_barrier.srcQueueFamilyIndex = queue_family_index_;
    image_memory_barrier.dstQueueFamilyIndex = queue_family_index_;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
    left_eye_desc_.image_layout = image_memory_barrier.newLayout;

    // Transition the depth buffer to VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL on first use
    if (left_eye_desc_.depth_stencil_image_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      image_memory_barrier.image = left_eye_desc_.depth_stencil_image;
      image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      image_memory_barrier.srcAccessMask = 0;
      image_memory_barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      image_memory_barrier.oldLayout = left_eye_desc_.depth_stencil_image_layout;
      image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
      left_eye_desc_.depth_stencil_image_layout = image_memory_barrier.newLayout;
    }

    // Start the renderpass
    VkRenderPassBeginInfo render_pass_begin_info = {};
    render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_begin_info.renderPass = left_eye_desc_.render_pass;
    render_pass_begin_info.framebuffer = left_eye_desc_.framebuffer;
    render_pass_begin_info.renderArea.offset.x = 0;
    render_pass_begin_info.renderArea.offset.y = 0;
    render_pass_begin_info.renderArea.extent.width = render_width_;
    render_pass_begin_info.renderArea.extent.height = render_height_;
    render_pass_begin_info.clearValueCount = 2;

    VkClearValue clear_values[2];
    clear_values[0].color.float32[0] = 0.0f;
    clear_values[0].color.float32[1] = 0.0f;
    clear_values[0].color.float32[2] = 0.0f;
    clear_values[0].color.float32[3] = 1.0f;
    clear_values[1].depthStencil.depth = 1.0f;
    clear_values[1].depthStencil.stencil = 0;
    render_pass_begin_info.pClearValues = &clear_values[0];
    vkCmdBeginRenderPass(current_command_buffer_.command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    renderScene(vr::Eye_Left);

    vkCmdEndRenderPass(current_command_buffer_.command_buffer);

    // Transition eye image to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL for display on the companion window
    image_memory_barrier.image = left_eye_desc_.image;
    image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_memory_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_memory_barrier.oldLayout = left_eye_desc_.image_layout;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
    left_eye_desc_.image_layout = image_memory_barrier.newLayout;

    //-----------//
    // Right Eye //
    //-----------//
    // Transition to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    image_memory_barrier.image = right_eye_desc_.image;
    image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_memory_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_memory_barrier.oldLayout = right_eye_desc_.image_layout;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
    right_eye_desc_.image_layout = image_memory_barrier.newLayout;

    // Transition the depth buffer to VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL on first use
    if (right_eye_desc_.depth_stencil_image_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      image_memory_barrier.image = right_eye_desc_.depth_stencil_image;
      image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      image_memory_barrier.srcAccessMask = 0;
      image_memory_barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      image_memory_barrier.oldLayout = right_eye_desc_.depth_stencil_image_layout;
      image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
      right_eye_desc_.depth_stencil_image_layout = image_memory_barrier.newLayout;
    }

    // Start the renderpass
    render_pass_begin_info.renderPass = right_eye_desc_.render_pass;
    render_pass_begin_info.framebuffer = right_eye_desc_.framebuffer;
    render_pass_begin_info.pClearValues = &clear_values[0];
    vkCmdBeginRenderPass(current_command_buffer_.command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    renderScene(vr::Eye_Right);

    vkCmdEndRenderPass(current_command_buffer_.command_buffer);

    // Transition eye image to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL for display on the companion window
    image_memory_barrier.image = right_eye_desc_.image;
    image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_memory_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_memory_barrier.oldLayout = right_eye_desc_.image_layout;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
    right_eye_desc_.image_layout = image_memory_barrier.newLayout;
  }

  //---------------------------------------------------------------------------
  // Purpose: Render for Companion window
  //---------------------------------------------------------------------------
  void renderCompanionWindow() {
    if (vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, swapchain_semaphores_[frame_index_], VK_NULL_HANDLE, &current_swapchain_image_) != VK_SUCCESS) {
      return;
    }

    // Transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL for rendering
    VkImageMemoryBarrier image_memory_barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    image_memory_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    image_memory_barrier.image = swapchain_images_[current_swapchain_image_];
    image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_memory_barrier.subresourceRange.baseMipLevel = 0;
    image_memory_barrier.subresourceRange.levelCount = 1;
    image_memory_barrier.subresourceRange.baseArrayLayer = 0;
    image_memory_barrier.subresourceRange.layerCount = 1;
    image_memory_barrier.srcQueueFamilyIndex = queue_family_index_;
    image_memory_barrier.dstQueueFamilyIndex = queue_family_index_;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);

    // Start the renderpass
    VkRenderPassBeginInfo render_pass_begin_info = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    render_pass_begin_info.renderPass = swapchain_render_pass_;
    render_pass_begin_info.framebuffer = swapchain_framebuffers_[current_swapchain_image_];
    render_pass_begin_info.renderArea.offset.x = 0;
    render_pass_begin_info.renderArea.offset.y = 0;
    render_pass_begin_info.renderArea.extent.width = companion_window_width_;
    render_pass_begin_info.renderArea.extent.height = companion_window_height_;
    VkClearValue clear_values[1];
    clear_values[0].color.float32[0] = 0.0f;
    clear_values[0].color.float32[1] = 0.0f;
    clear_values[0].color.float32[2] = 0.0f;
    clear_values[0].color.float32[3] = 1.0f;
    render_pass_begin_info.clearValueCount = _countof(clear_values);
    render_pass_begin_info.pClearValues = &clear_values[0];
    vkCmdBeginRenderPass(current_command_buffer_.command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    // Set viewport/scissor
    VkViewport viewport = { 0.0f, 0.0f, (float)companion_window_width_, (float)companion_window_height_, 0.0f, 1.0f };
    vkCmdSetViewport(current_command_buffer_.command_buffer, 0, 1, &viewport);
    VkRect2D scissor = { 0, 0, companion_window_width_, companion_window_height_ };
    vkCmdSetScissor(current_command_buffer_.command_buffer, 0, 1, &scissor);

    // Bind the pipeline and descriptor set
    vkCmdBindPipeline(current_command_buffer_.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[PSO_COMPANION]);
    vkCmdBindDescriptorSets(current_command_buffer_.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &descriptor_sets_[DESCRIPTOR_SET_COMPANION_LEFT_TEXTURE], 0, nullptr);

    // Draw left eye texture to companion window
    VkDeviceSize nOffsets[1] = { 0 };
    vkCmdBindVertexBuffers(current_command_buffer_.command_buffer, 0, 1, &companion_window_vertex_buffer_, &nOffsets[0]);
    vkCmdBindIndexBuffer(current_command_buffer_.command_buffer, companion_window_index_buffer_, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(current_command_buffer_.command_buffer, companion_window_index_size_ / 2, 1, 0, 0, 0);

    // Draw right eye texture to companion window
    vkCmdBindDescriptorSets(current_command_buffer_.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &descriptor_sets_[DESCRIPTOR_SET_COMPANION_RIGHT_TEXTURE], 0, nullptr);
    vkCmdDrawIndexed(current_command_buffer_.command_buffer, companion_window_index_size_ / 2, 1, (companion_window_index_size_ / 2), 0, 0);

    // End the renderpass
    vkCmdEndRenderPass(current_command_buffer_.command_buffer);

    // Transition the swapchain image to PRESENT_SRC for presentation
    image_memory_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);

    // Transition both of the eye textures to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL for SteamVR which requires this layout for submit
    image_memory_barrier.image = left_eye_desc_.image;
    image_memory_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    image_memory_barrier.oldLayout = left_eye_desc_.image_layout;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
    left_eye_desc_.image_layout = image_memory_barrier.newLayout;

    image_memory_barrier.image = right_eye_desc_.image;
    vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
    right_eye_desc_.image_layout = image_memory_barrier.newLayout;
  }

  //---------------------------------------------------------------------------
  // Purpose: Update HMD Pose. From hellovr_vulkan in openVR SDK
  //---------------------------------------------------------------------------
  void updateHMDMatrixPose() {
    if (!hmd_)
      return;

    vr::VRCompositor()->WaitGetPoses(tracked_device_pose_, vr::k_unMaxTrackedDeviceCount, NULL, 0);

    valid_pose_count_ = 0;
    str_pose_classes_ = "";
    for (int device = 0; device < vr::k_unMaxTrackedDeviceCount; ++device) {
      if (tracked_device_pose_[device].bPoseIsValid) {
        valid_pose_count_++;
        mat4_device_pose_[device] = convertSteamVRMatrixToMatrix4(tracked_device_pose_[device].mDeviceToAbsoluteTracking);
        if (dev_class_char[device] == 0) {
          switch (hmd_->GetTrackedDeviceClass(device)) {
          case vr::TrackedDeviceClass_Controller:        dev_class_char[device] = 'C'; break;
          case vr::TrackedDeviceClass_HMD:               dev_class_char[device] = 'H'; break;
          case vr::TrackedDeviceClass_Invalid:           dev_class_char[device] = 'I'; break;
          case vr::TrackedDeviceClass_GenericTracker:    dev_class_char[device] = 'G'; break;
          case vr::TrackedDeviceClass_TrackingReference: dev_class_char[device] = 'T'; break;
          default:                                       dev_class_char[device] = '?'; break;
          }
        }
        str_pose_classes_ += dev_class_char[device];
      }
    }

    if (tracked_device_pose_[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
      mat4_hmd_pose_ = mat4_device_pose_[vr::k_unTrackedDeviceIndex_Hmd];
      mat4_hmd_pose_.invert();
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: Rendering the scene with respect to the eye position
  //---------------------------------------------------------------------------
  void renderScene(vr::Hmd_Eye nEye) {
    vkCmdBindPipeline(current_command_buffer_.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[PSO_SCENE]);

    // Update the persistently mapped pointer to the CB data with the latest matrix
    memcpy(scene_constant_buffer_data_[nEye], getCurrentViewProjectionMatrix(nEye).get(), sizeof(Matrix4));

    vkCmdBindDescriptorSets(current_command_buffer_.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &descriptor_sets_[DESCRIPTOR_SET_LEFT_EYE_SCENE + nEye], 0, nullptr);

    // Draw
    VkDeviceSize nOffsets[1] = { 0 };
    vkCmdBindVertexBuffers(current_command_buffer_.command_buffer, 0, 1, &scene_vertex_buffer_, &nOffsets[0]);
    vkCmdDraw(current_command_buffer_.command_buffer, vert_count_, 1, 0, 0);
  }



  //-------------------------------------------------------------------------//
  //-------------------------------------------------------------------------//
  //                           HELPER FUNCTIONS                              //
  //-------------------------------------------------------------------------//
  //-------------------------------------------------------------------------//

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
  // Purpose: convert SteamVR Matrix To Matrix 4
  //---------------------------------------------------------------------------
  Matrix4 convertSteamVRMatrixToMatrix4(const vr::HmdMatrix34_t &matPose) {
    Matrix4 matrixObj(
      matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0,
      matPose.m[0][1], matPose.m[1][1], matPose.m[2][1], 0.0,
      matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0,
      matPose.m[0][3], matPose.m[1][3], matPose.m[2][3], 1.0f
    );
    return matrixObj;
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
  // Purpose: helper function to create buffers
  //---------------------------------------------------------------------------
  void createAndFillBuffer(const void *buffer_data, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
    // create buffer
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size; // buffer size in bytes
    buffer_info.usage = usage;

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

    if (buffer_data != nullptr) {
      void *data;
      vkMapMemory(device_, buffer_memory, 0, VK_WHOLE_SIZE, 0, &data);
      memcpy(data, buffer_data, size);
      vkUnmapMemory(device_, buffer_memory);

      VkMappedMemoryRange memory_range = {};
      memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      memory_range.memory = buffer_memory;
      memory_range.size = VK_WHOLE_SIZE;
      vkFlushMappedMemoryRanges(device_, 1, &memory_range);
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: create an image view
  //---------------------------------------------------------------------------
  VkImageView createImageView(VkImage image, VkImageView image_view, VkFormat format, VkImageAspectFlags aspect_flags) {
    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.flags = 0;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D; // allows you to treat images as 1D, 2D, 3D textures or cube maps
    view_info.format = format;
    view_info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
    view_info.subresourceRange.aspectMask = aspect_flags;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device_, &view_info, nullptr, &image_view) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create texture image view!");
    }
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
    const char* layer_prefix, const char* msg, void* user_data) {
    std::cerr << "Validation layer: " << msg << std::endl;

    return VK_FALSE;
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

    vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);

    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
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
  // Purpose: Get available command buffer or create a new one if none...
  // ...available. Also associate fence with the command buffer
  //---------------------------------------------------------------------------
  VulkanCommandBuffer_t getCommandBuffer() {
    VulkanCommandBuffer_t command_buffer;
    if (command_buffers_.size() > 0) {
      // If the fence associated with the command buffer has finished, reset it and return it
      if (vkGetFenceStatus(device_, command_buffers_.back().fence) == VK_SUCCESS) {
        VulkanCommandBuffer_t *p_cmd_buffer = &command_buffers_.back();
        command_buffer.command_buffer = p_cmd_buffer->command_buffer;
        command_buffer.fence = p_cmd_buffer->fence;

        vkResetCommandBuffer(command_buffer.command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
        vkResetFences(device_, 1, &command_buffer.fence);
        command_buffers_.pop_back();
        return command_buffer;
      }
    }
    // Create a new command buffer with the associated fence
    VkCommandBufferAllocateInfo command_buffer_alloc_info = {};
    command_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_alloc_info.commandBufferCount = 1;
    command_buffer_alloc_info.commandPool = command_pool_;
    command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    vkAllocateCommandBuffers(device_, &command_buffer_alloc_info, &command_buffer.command_buffer);

    VkFenceCreateInfo fence_create_info = {};
    fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device_, &fence_create_info, nullptr, &command_buffer.fence);
    return command_buffer;

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
          const char* temp = _strdup(curExtStr.c_str());
          extensions.push_back(temp);
          curExtStr.clear();
        }
        else {
          curExtStr += pExtensionStr[nIndex];
        }
        nIndex++;
      }
      if (curExtStr.size() > 0) {
        const char* temp = _strdup(curExtStr.c_str());
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
  // Purpose: check for stencil component
  //---------------------------------------------------------------------------
  bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
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

  //-------------------------------------------------------------------------//
  //-------------------------------------------------------------------------//
  //                       VIRTUAL REALITY FUNCTIONS                         //
  //-------------------------------------------------------------------------//
  //-------------------------------------------------------------------------//

  //-----------------------------------------------------------------------------
  // Purpose: generate next level mipmap for an RGBA image, from hellovr_vulkan...
  // ...from openVR sdk sample codes.
  //-----------------------------------------------------------------------------
  void GenMipMapRGBA(const uint8_t *pSrc, uint8_t *pDst, int nSrcWidth, int nSrcHeight, int *pDstWidthOut, int *pDstHeightOut) {
    *pDstWidthOut = nSrcWidth / 2;
    if (*pDstWidthOut <= 0) {
      *pDstWidthOut = 1;
    }
    *pDstHeightOut = nSrcHeight / 2;
    if (*pDstHeightOut <= 0) {
      *pDstHeightOut = 1;
    }

    for (int y = 0; y < *pDstHeightOut; y++) {
      for (int x = 0; x < *pDstWidthOut; x++) {
        int nSrcIndex[4];
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float a = 0.0f;

        nSrcIndex[0] = (((y * 2) * nSrcWidth) + (x * 2)) * 4;
        nSrcIndex[1] = (((y * 2) * nSrcWidth) + (x * 2 + 1)) * 4;
        nSrcIndex[2] = ((((y * 2) + 1) * nSrcWidth) + (x * 2)) * 4;
        nSrcIndex[3] = ((((y * 2) + 1) * nSrcWidth) + (x * 2 + 1)) * 4;

        // Sum all pixels
        for (int nSample = 0; nSample < 4; nSample++) {
          r += pSrc[nSrcIndex[nSample]];
          g += pSrc[nSrcIndex[nSample] + 1];
          b += pSrc[nSrcIndex[nSample] + 2];
          a += pSrc[nSrcIndex[nSample] + 3];
        }

        // Average results
        r /= 4.0;
        g /= 4.0;
        b /= 4.0;
        a /= 4.0;

        // Store resulting pixels
        pDst[(y * (*pDstWidthOut) + x) * 4] = (uint8_t)(r);
        pDst[(y * (*pDstWidthOut) + x) * 4 + 1] = (uint8_t)(g);
        pDst[(y * (*pDstWidthOut) + x) * 4 + 2] = (uint8_t)(b);
        pDst[(y * (*pDstWidthOut) + x) * 4 + 3] = (uint8_t)(a);
      }
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: Get updated HMD pose 
  //---------------------------------------------------------------------------
  Matrix4 getCurrentViewProjectionMatrix(vr::Hmd_Eye eye) {
    Matrix4 mat4MVP;
    if (eye == vr::Eye_Left) {
      mat4MVP = mat4_proj_left_ * mat4_eye_pos_left_ * mat4_hmd_pose_;
    }
    else if (eye == vr::Eye_Right) {
      mat4MVP = mat4_proj_right_ * mat4_eye_pos_right_ *  mat4_hmd_pose_;
    }

    return mat4MVP;
  }

  //---------------------------------------------------------------------------
  // Purpose: [VR] Initialize the VR custom compositor
  //---------------------------------------------------------------------------
  void initVRCompositor() {
    vr::EVRInitError error = vr::VRInitError_None;

    if (!vr::VRCompositor()) {
      throw std::runtime_error("Compositor initialization failed.");
    }
  }

  //---------------------------------------------------------------------------
  // Purpose: [VR] Set eye and projection cameras
  //---------------------------------------------------------------------------
  void setupCameras() {
    mat4_proj_left_ = getHMDMatrixProjectionEye(vr::Eye_Left);
    mat4_proj_right_ = getHMDMatrixProjectionEye(vr::Eye_Right);
    mat4_eye_pos_left_ = getHMDMatrixPoseEye(vr::Eye_Left);
    mat4_eye_pos_right_ = getHMDMatrixPoseEye(vr::Eye_Right);
  }

  //---------------------------------------------------------------------------
  // Purpose: [VR] Get the projection matrix for the HMD eye
  //---------------------------------------------------------------------------
  Matrix4 getHMDMatrixProjectionEye(vr::Hmd_Eye eye) {
    if (!hmd_) { return Matrix4(); }
    vr::HmdMatrix44_t mat = hmd_->GetProjectionMatrix(eye, near_clip_, far_clip_);

    return Matrix4(
      mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
      mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
      mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
      mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
    );
  }

  //---------------------------------------------------------------------------
  // Purpose: [VR] Get the HMD eye position
  //---------------------------------------------------------------------------
  Matrix4 getHMDMatrixPoseEye(vr::Hmd_Eye eye) {
    if (!hmd_) { return Matrix4(); }
    vr::HmdMatrix34_t mat_eye_right = hmd_->GetEyeToHeadTransform(eye);
    Matrix4 matrix_obj(
      mat_eye_right.m[0][0], mat_eye_right.m[1][0], mat_eye_right.m[2][0], 0.0,
      mat_eye_right.m[0][1], mat_eye_right.m[1][1], mat_eye_right.m[2][1], 0.0,
      mat_eye_right.m[0][2], mat_eye_right.m[1][2], mat_eye_right.m[2][2], 0.0,
      mat_eye_right.m[0][3], mat_eye_right.m[1][3], mat_eye_right.m[2][3], 1.0f
    );
    return matrix_obj.invert();
  }

  //---------------------------------------------------------------------------
  // Purpose: [VR] Set up render targets
  //---------------------------------------------------------------------------
  void setupStereoRenderTargets() {
    if (!hmd_) {
      throw std::runtime_error("Failed to set up render targets");
    }

    hmd_->GetRecommendedRenderTargetSize(&render_width_, &render_height_);
    createFrameBufferDesc(render_width_, render_height_, left_eye_desc_);
    createFrameBufferDesc(render_width_, render_height_, right_eye_desc_);
  }

  //---------------------------------------------------------------------------
  // Purpose: [VR] Create Frame Buffer Descriptions for HMD Eyes
  //---------------------------------------------------------------------------
  void createFrameBufferDesc(int nWidth, int nHeight, FramebufferDesc &framebufferDesc) {
    //---------------------------//
    //    Create color target    //
    //---------------------------//
    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.extent.width = nWidth;
    image_create_info.extent.height = nHeight;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.format = VK_FORMAT_R8G8B8A8_SRGB;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.samples = VK_SAMPLE_COUNT_4_BIT;
    image_create_info.usage = (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    image_create_info.flags = 0;

    vkCreateImage(device_, &image_create_info, nullptr, &framebufferDesc.image);

    VkMemoryRequirements memory_requirements = {};
    vkGetImageMemoryRequirements(device_, framebufferDesc.image, &memory_requirements);

    VkMemoryAllocateInfo memory_allocate_info = {};
    memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_allocate_info.allocationSize = memory_requirements.size;
    memory_allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkAllocateMemory(device_, &memory_allocate_info, nullptr, &framebufferDesc.device_memory);

    vkBindImageMemory(device_, framebufferDesc.image, framebufferDesc.device_memory, 0);

    VkImageViewCreateInfo image_view_create_info = {};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.flags = 0;
    image_view_create_info.image = framebufferDesc.image;
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = image_create_info.format;
    image_view_create_info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = 0;
    image_view_create_info.subresourceRange.levelCount = 1;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount = 1;

    vkCreateImageView(device_, &image_view_create_info, nullptr, &framebufferDesc.image_view);

    //-----------------------------------//
    //    Create depth/stencil target    //
    //-----------------------------------//
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = VK_FORMAT_D32_SFLOAT;
    image_create_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    vkCreateImage(device_, &image_create_info, nullptr, &framebufferDesc.depth_stencil_image);

    vkGetImageMemoryRequirements(device_, framebufferDesc.depth_stencil_image, &memory_requirements);
    memory_allocate_info.allocationSize = memory_requirements.size;
    memory_allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device_, &memory_allocate_info, nullptr, &framebufferDesc.depth_stencil_device_memory);
    vkBindImageMemory(device_, framebufferDesc.depth_stencil_image, framebufferDesc.depth_stencil_device_memory, 0);

    image_view_create_info.image = framebufferDesc.depth_stencil_image;
    image_view_create_info.format = image_create_info.format;
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vkCreateImageView(device_, &image_view_create_info, nullptr, &framebufferDesc.depth_stencil_image_view);

    // Create a renderpass
    uint32_t total_attachments = 2;
    VkAttachmentReference attachment_references[2];
    attachment_references[0].attachment = 0;
    attachment_references[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment_references[1].attachment = 1;
    attachment_references[1].layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription attachment_descs[2];
    attachment_descs[0].format = VK_FORMAT_R8G8B8A8_SRGB;
    attachment_descs[0].samples = image_create_info.samples;
    attachment_descs[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment_descs[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment_descs[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment_descs[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment_descs[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment_descs[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment_descs[0].flags = 0;

    attachment_descs[1].format = VK_FORMAT_D32_SFLOAT;
    attachment_descs[1].samples = image_create_info.samples;
    attachment_descs[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment_descs[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment_descs[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment_descs[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment_descs[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachment_descs[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachment_descs[1].flags = 0;

    VkSubpassDescription subpass_create_info = {};
    subpass_create_info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_create_info.flags = 0;
    subpass_create_info.inputAttachmentCount = 0;
    subpass_create_info.pInputAttachments = NULL;
    subpass_create_info.colorAttachmentCount = 1;
    subpass_create_info.pColorAttachments = &attachment_references[0];
    subpass_create_info.pResolveAttachments = NULL;
    subpass_create_info.pDepthStencilAttachment = &attachment_references[1];
    subpass_create_info.preserveAttachmentCount = 0;
    subpass_create_info.pPreserveAttachments = NULL;

    VkRenderPassCreateInfo render_pass_create_info = {};
    render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_create_info.flags = 0;
    render_pass_create_info.attachmentCount = 2;
    render_pass_create_info.pAttachments = &attachment_descs[0];
    render_pass_create_info.subpassCount = 1;
    render_pass_create_info.pSubpasses = &subpass_create_info;
    render_pass_create_info.dependencyCount = 0;
    render_pass_create_info.pDependencies = NULL;

    vkCreateRenderPass(device_, &render_pass_create_info, NULL, &framebufferDesc.render_pass);

    // Create the framebuffer
    VkImageView attachments[2] = { framebufferDesc.image_view, framebufferDesc.depth_stencil_image_view };
    VkFramebufferCreateInfo framebuffer_create_info = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    framebuffer_create_info.renderPass = framebufferDesc.render_pass;
    framebuffer_create_info.attachmentCount = 2;
    framebuffer_create_info.pAttachments = &attachments[0];
    framebuffer_create_info.width = nWidth;
    framebuffer_create_info.height = nHeight;
    framebuffer_create_info.layers = 1;
    vkCreateFramebuffer(device_, &framebuffer_create_info, NULL, &framebufferDesc.framebuffer);

    framebufferDesc.image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    framebufferDesc.depth_stencil_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  }

  //---------------------------------------------------------------------------
  // Purpose: [VR] Set up the companion window
  //---------------------------------------------------------------------------
  void setupCompanionWindow() {
    if (!hmd_) {
      return;
    }

    std::vector<VertexDataWindow> verts;

    // left eye verts
    verts.push_back(VertexDataWindow(Vector2(-1, -1), Vector2(0, 1)));
    verts.push_back(VertexDataWindow(Vector2(0, -1), Vector2(1, 1)));
    verts.push_back(VertexDataWindow(Vector2(-1, 1), Vector2(0, 0)));
    verts.push_back(VertexDataWindow(Vector2(0, 1), Vector2(1, 0)));

    // right eye verts
    verts.push_back(VertexDataWindow(Vector2(0, -1), Vector2(0, 1)));
    verts.push_back(VertexDataWindow(Vector2(1, -1), Vector2(1, 1)));
    verts.push_back(VertexDataWindow(Vector2(0, 1), Vector2(0, 0)));
    verts.push_back(VertexDataWindow(Vector2(1, 1), Vector2(1, 0)));

    createAndFillBuffer(&verts[0], sizeof(VertexDataWindow) * verts.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, companion_window_vertex_buffer_, companion_window_vertex_buffer_memory_);

    uint16_t indices[] = { 0, 1, 3,   0, 3, 2,   4, 5, 7,   4, 7, 6 };
    companion_window_index_size_ = _countof(indices);
    createAndFillBuffer(&indices[0], sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
      companion_window_index_buffer_, companion_window_index_buffer_memory_);

    // Transition all of the swapchain images to PRESENT_SRC so they are ready for presentation
    for (size_t swapchain_image = 0; swapchain_image < swapchain_images_.size(); swapchain_image++) {
      VkImageMemoryBarrier image_memory_barrier = {};
      image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      image_memory_barrier.srcAccessMask = 0;
      image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      image_memory_barrier.image = swapchain_images_[swapchain_image];
      image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      image_memory_barrier.subresourceRange.baseMipLevel = 0;
      image_memory_barrier.subresourceRange.levelCount = 1;
      image_memory_barrier.subresourceRange.baseArrayLayer = 0;
      image_memory_barrier.subresourceRange.layerCount = 1;
      image_memory_barrier.srcQueueFamilyIndex = queue_family_index_;
      image_memory_barrier.dstQueueFamilyIndex = queue_family_index_;
      vkCmdPipelineBarrier(current_command_buffer_.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &image_memory_barrier);
    }
  }
};

//---------------------------------------------------------------------------
// Purpose: 
//---------------------------------------------------------------------------
int main() {
  VulkanVRApplication app;

  try {
    app.run();
  }
  catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}