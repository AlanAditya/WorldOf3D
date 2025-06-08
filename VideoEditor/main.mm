//
//  main.cpp
//  VideoEditor
//
//  Created by Manoj Kumar on 20/01/25.
//
#define GL_SILENCE_DEPRECATION

#include <iostream>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libswscale/swscale.h>
}
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
//#import <GLFW/glfw3.h>
#import <OpenGL/OpenGL.h>
#import <GLKit/GLKit.h>
#import <AppKit/AppKit.h>
#import <Cocoa/Cocoa.h>

int cx, cy;
GLuint glposition;
GLuint glright;
GLuint glforward;
GLuint glup;
GLuint glorigin;
GLuint glx;
GLuint gly;
GLuint gllen;
GLuint gl;
auto date = std::chrono::system_clock::now();
int md = 0;
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(date.time_since_epoch());
long long t2,t1 = duration.count();
int mx = 0, my = 0, mx1 = 0, my1 = 0, lasttimen = 0;
int ml = 0, mr = 0, mm = 0;
float len = 1.6;
float ang1 = 2.8;
float ang2 = 0.4;
float cenx = 0.0;
float ceny = 0.0;
float cenz = 0.0;
const char* vertexShaderSource = R"(

#version 110 
attribute vec4 position;
varying vec3 dir, localdir;
uniform vec3 right, forward, up, origin;
uniform float x,y;
void main() {
    gl_Position = position;
    gl_Position = position; 
    dir = forward + right * position.x*x + up * position.y*y;
    localdir.x = position.x*x;
    localdir.y = position.y*y;
    localdir.z = -1.0;
}
)";


// Fragment Shader source code
const char* fragmentShaderSource = R"(
#version 110
#define PI 3.14159265358979324
#define M_L 0.3819660113
#define M_R 0.6180339887
#define MAXR 8
#define SOLVER 8
float kernal(vec3 ver);
uniform vec3 right, forward, up, origin;
varying vec3 dir, localdir;
uniform float len;
vec3 ver;
int sign;
float v, v1, v2;
float r1, r2, r3, r4, m1, m2, m3, m4;
vec3 n, reflect;
const float step = 0.002;
vec3 color;
void main() {
   color.r=0.0;
   color.g=0.0;
   color.b=0.0;
   sign=0;
   v1 = kernal(origin + dir * (step*len));
   v2 = kernal(origin);
   for (int k = 2; k < 1002; k++) {
      ver = origin + dir * (step*len*float(k));
      v = kernal(ver);
      if (v > 0.0 && v1 < 0.0) {
         r1 = step * len*float(k - 1);
         r2 = step * len*float(k);
         m1 = kernal(origin + dir * r1);
         m2 = kernal(origin + dir * r2);
         for (int l = 0; l < SOLVER; l++) {
            r3 = r1 * 0.5 + r2 * 0.5;
            m3 = kernal(origin + dir * r3);
            if (m3 > 0.0) {
               r2 = r3;
               m2 = m3;
            }
            else {
               r1 = r3;
               m1 = m3;
            }
         }
         if (r3 < 2.0 * len) {
            sign=1;
            break;
         }
      }
      if (v < v1&&v1>v2&&v1 < 0.0 && (v1*2.0 > v || v1 * 2.0 > v2)) {
         r1 = step * len*float(k - 2);
         r2 = step * len*(float(k) - 2.0 + 2.0*M_L);
         r3 = step * len*(float(k) - 2.0 + 2.0*M_R);
         r4 = step * len*float(k);
         m2 = kernal(origin + dir * r2);
         m3 = kernal(origin + dir * r3);
         for (int l = 0; l < MAXR; l++) {
            if (m2 > m3) {
               r4 = r3;
               r3 = r2;
               r2 = r4 * M_L + r1 * M_R;
               m3 = m2;
               m2 = kernal(origin + dir * r2);
            }
            else {
               r1 = r2;
               r2 = r3;
               r3 = r4 * M_R + r1 * M_L;
               m2 = m3;
               m3 = kernal(origin + dir * r3);
            }
         }
         if (m2 > 0.0) {
            r1 = step * len*float(k - 2);
            r2 = r2;
            m1 = kernal(origin + dir * r1);
            m2 = kernal(origin + dir * r2);
            for (int l = 0; l < SOLVER; l++) {
               r3 = r1 * 0.5 + r2 * 0.5;
               m3 = kernal(origin + dir * r3);
               if (m3 > 0.0) {
                  r2 = r3;
                  m2 = m3;
               }
               else {
                  r1 = r3;
                  m1 = m3;
               }
            }
            if (r3 < 2.0 * len&&r3> step*len) {
                sign=1;
               break;
            }
         }
         else if (m3 > 0.0) {
            r1 = step * len*float(k - 2);
            r2 = r3;
            m1 = kernal(origin + dir * r1);
            m2 = kernal(origin + dir * r2);
            for (int l = 0; l < SOLVER; l++) {
               r3 = r1 * 0.5 + r2 * 0.5;
               m3 = kernal(origin + dir * r3);
               if (m3 > 0.0) {
                  r2 = r3;
                  m2 = m3;
               }
               else {
                  r1 = r3;
                  m1 = m3;
               }
            }
            if (r3 < 2.0 * len&&r3> step*len) {
               sign=1;
               break;
            }
         }
      }
      v2 = v1;
      v1 = v;
   }
   if (sign==1) {
      ver = origin + dir*r3 ;
      r1=ver.x*ver.x+ver.y*ver.y+ver.z*ver.z;
      n.x = kernal(ver - right * (r3*0.00025)) - kernal(ver + right * (r3*0.00025));
      n.y = kernal(ver - up * (r3*0.00025)) - kernal(ver + up * (r3*0.00025));
      n.z = kernal(ver + forward * (r3*0.00025)) - kernal(ver - forward * (r3*0.00025));
      r3 = n.x*n.x+n.y*n.y+n.z*n.z;
      n = n * (1.0 / sqrt(r3));
      ver = localdir;
      r3 = ver.x*ver.x+ver.y*ver.y+ver.z*ver.z;
      ver = ver * (1.0 / sqrt(r3));
      reflect = n * (-2.0*dot(ver, n)) + ver;
      r3 = reflect.x*0.276+reflect.y*0.920+reflect.z*0.276;
      r4 = n.x*0.276+n.y*0.920+n.z*0.276;
      r3 = max(0.0,r3);
      r3 = r3 * r3*r3*r3;
      r3 = r3 * 0.45 + r4 * 0.25 + 0.3;
      n.x = sin(r1*10.0)*0.5+0.5;
      n.y = sin(r1*10.0+2.05)*0.5+0.5;
      n.z = sin(r1*10.0-2.05)*0.5+0.5;
      color = n*r3;
   }
   gl_FragColor = vec4(color.x, color.y, color.z, 1.0);
}

float kernal(vec3 ver){
   vec3 a;
   float b,c,d,e;
   a=ver;
   for(int i=0;i<5;i++){
       b=length(a);
       c=atan(a.y,a.x)*8.0;
       e=1.0/b;
       d=acos(a.z/b)*8.0;
       b=pow(b,8.0);
       a=vec3(b*sin(d)*cos(c),b*sin(d)*sin(c),b*cos(d))+ver;
       if(b>6.0){
           break;
       }
   }
   return 4.0-a.x*a.x-a.y*a.y-a.z*a.z;
}
)";

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check for errors
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Error: " << infoLog << std::endl;
    }
    return shader;
}

// Callback function for window resize
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void draw() {
    
    date = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(date.time_since_epoch());
    long long t2 = duration.count();
    //ang1 += (t2 - t1)*0.001;
    t1 = t2;
    //ang1 += 0.1;
    glUniform1f(glx, cx * 2.0 / (cx + cy));
    glUniform1f(gly, cy * 2.0 / (cx + cy));
    glUniform1f(gllen, len);
    glUniform3f(glorigin, len * cos(ang1) * cos(ang2) + cenx, len * sin(ang2) + ceny, len * sin(ang1) * cos(ang2) + cenz);
    glUniform3f(glright, sin(ang1), 0, -cos(ang1));
    glUniform3f(glup, -sin(ang2) * cos(ang1), cos(ang2), -sin(ang2) * sin(ang1));
    glUniform3f(glforward, -cos(ang1) * cos(ang2), -sin(ang2), -sin(ang1) * cos(ang2));
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glFinish();
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (ml == 1) {
        ang1 += (xpos - mx) * 0.002;
        ang2 += (ypos - my) * 0.002;
        if (xpos != mx || ypos != my) {
            mm = 1;
        }
    }
    if (mr == 1) {
        float l = len * 4.0f / (cx + cy);
        cenx += l * (-(xpos - mx) * sin(ang1) - (ypos - my) * sin(ang2) * cos(ang1));
        ceny += l * ((ypos - my) * cos(ang2));
        cenz += l * ((xpos - mx) * cos(ang1) - (ypos - my) * sin(ang2) * sin(ang1));
        if (xpos != mx || ypos != my) {
            mm = 1;
        }
    }
    mx = xpos;
    my = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            ml = 1;
            mm = 0;
        } else if (action == GLFW_RELEASE) {
            ml = 0;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mr = 1;
            mm = 0;
        } else if (action == GLFW_RELEASE) {
            mr = 0;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    len *= exp(-0.001 * yoffset);
}

int main() {
    sleep(1);
    
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    // Create GLFW window
    if (glfwGetCurrentContext() != nullptr) {
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(800, 600, "GLFW Window", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the OpenGL context current
    
    glfwMakeContextCurrent(window);

    // Set framebuffer resize callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    
    float positions[] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0};
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // Bind VAO
    glBindVertexArray(VAO);

    // Bind and fill VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

    // Define vertex attribute layout
    glVertexAttribPointer(glposition, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(glposition);
    
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);
    
    glposition = glGetAttribLocation(shaderProgram, "position");
    glright = glGetUniformLocation(shaderProgram, "right");
    glforward = glGetUniformLocation(shaderProgram, "forward");
    glup = glGetUniformLocation(shaderProgram, "up");
    glorigin = glGetUniformLocation(shaderProgram, "origin");
    glx = glGetUniformLocation(shaderProgram, "x");
    gly = glGetUniformLocation(shaderProgram, "y");
    gllen = glGetUniformLocation(shaderProgram, "len");
    
    
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader Program Linking Error: " << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwGetWindowSize(window, &cx, &cy);
        // Process events
        glfwPollEvents();

        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        draw();
        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
