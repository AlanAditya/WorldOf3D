//
//  main.cpp
//  Algorithms
//
//  Created by Manoj Kumar on 01/11/24.
//

#include <iostream>
#include "Fibonacci.hpp"
#include "HELLLLLL.hpp"
//#include <Carbon/Carbon.h>

#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>
#include "Header.h"
#include <objc/objc.h>
//#include <objc/objc-runtime.h>

//#include <AVFoundation/AVFoundation.h>
//#include <Foundation/NSString.h>
#include <objc/runtime.h>
#include "objc/message.h"
#include <objc/NSObjCRuntime.h>

void Problem1();
void Problem2();
void Problem3();
void Problem4();
void Problem5();
void Problem6();
void Problem7();
void Problem8();
void Problem9();

//int main(int argc, const char * argv[]) {
//    // insert code here...
////    std::cout << "Hello, World!\n";
//    Problem9();
//    return 0;
//}


void Problem1() {
    int m, n;
    
    std::cout << "Enter initial number of elements:";
    std::cin >> n;
    
    int *array1 = new int[n];
    
    std::cout << "Enter additional elements: ";
    for (int i = 0; i < n; i++) {
        std::cin >> array1[i];
    }
    
    std::cout << "Enter additional number of elements: ";
    std::cin >> m;
    
    int* array2 = new int[m+n];
    
    for (int i=0; i<n; i++) {
        array2[i] = array1[i];
    }
    
    std::cout << "Enter additional elements: ";
    for (int i = 0; i < m; i++) {
        std::cin >> array2[n+i];
    }
    
    std::cout << "Finnal Array: ";
    for (int i = 0; i < n+m; i++) {
        std::cout << array2[i] << " ";
    }
    std::cout << std::endl;
}

void Problem3() {
    int n;
    std::cout << "Enter number of elements: ";
    std::cin >> n;
    int* array = new int[n];
    
    int nonZero = 0;
    std::cout << "Enter elements: ";
    for (int i = 0; i < n; i++) {
        int k;
        std::cin >> k;
        if (k==0) {
            array[n-(i-nonZero)-1] = k;
        } else {
            array[nonZero] = k;
            nonZero++;
        }
    }
    std::cout << "Array after moving zeros to end: ";
    for (int i = 0; i < n; i++) {
        std::cout << array[i] << " ";
    }
}

void Problem2() {
    int size = 3;
    int minROW[3];
    int maxCOL[3];
    
    int matrix[size][size];
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int k;
            std::cin >> k;
            matrix[i][j] = k;
            if (i == 0) { maxCOL[j] = i; }
            else if (k >  matrix[maxCOL[j]][j]) { maxCOL[j] = i; }
            
            if (j == 0) { minROW[i] = j; }
            else if (k < matrix[i][minROW[i]]) { minROW[i] = j; }
        }
    }
    for (int i=0; i< 3; i++) {
        std::cout << maxCOL[i];
    }
    std::cout << std::endl;
    for (int i=0; i< 3; i++) {
        std::cout << minROW[i];
    }
    std::cout << std::endl;
    
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            if (c == minROW[r] && r == maxCOL[c]) {
                std::cout << "found: " << matrix[r][c];
            }
        }
    }
    
}

void Problem4() {
    int size = 4;
    
    int matrix[size][size];
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int k;
            std::cin >> k;
            matrix[i][j] = k;
        }
    }
    for (int i = 0; i < size; i++) {
        std::cout << matrix[0][i] << " ";
    }
    
    for (int i = 1; i < size; i++) {
        std::cout << matrix[i][size-1] << " ";
    }

    for (int i = 0; i < size-1; i++) {
        std::cout << matrix[size-1][size - i -2] << " ";
    }
    for (int i = 0; i < size-2; i++) {
        std::cout << matrix[size - i -2][0] << " ";
    }
}

void Problem5() {
    int m = 3;
    int n = 3;
    
    int matrix[m][n];
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int k;
            std::cin >> k;
            matrix[i][j] = k;
        }
    }
    for (int i =0 ; i < m; i++) {
        if (i % 2 == 0) {
            for (int j =0; j < n; j++) {
                std::cout << matrix[i][j] << " ";
            }
        } else {
            for (int j =n-1; j > -1; j--) {
                std::cout << matrix[i][j] << " ";
            }
        }
    }
        
}

void Problem6() {
    int n;
    int targetSum;
    std::cout << "Enter number of elements: ";
    std::cin >> n;
    
    int *array = new int[n];
    std::cout << "Enter elements: ";
    for (int i =0; i < n; i++) {
        std::cin >> array[i];
    }
    
    
    std::cout << "Enter target sum: ";
    std::cin >> targetSum;
    
    std::cout << "Unique pairs are: ";
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            if (array[i] + array[j] == targetSum) {
                std::cout << "(" << array[i] << ", " << array[j] << ")" << std::endl;
            }
        }
    }
}

void Problem7() {
    int n = 3;
    std::cout << "Enter the size of the lower triangular matrix: ";
    std::cin >> n;
    int *ODARRAY = new int[(n*(n+1))/2];
    
    std::cout << "Enter matrix elements row by row: \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int k;
            std::cin >> k;
            if (j <= i) {
                int S = (i*(i+1))/2;
                ODARRAY[S + j] = k;
            }
        }
    }
    
    std::cout << "The stored matrix is:" << std::endl;
    for (int i =0; i < n; i++) {
        int S = (i*(i+1))/2;
        for (int j = 0; j < i+1; j++) {
            std::cout << ODARRAY[S + j] << " ";
        }
        for (int j = 0; j < (n-i-1); j++) {
            std::cout << 0  << " ";
        }
        std::cout << std::endl;
    }
        
}

void Problem8() {
    int n;
    std::cout << "Enter number of elements: ";
    std::cin >> n;
    int sumRow[3] = {0, 0, 0};
    int sumCol[3] = {0, 0, 0};
    int sumDiag[2] = {0, 0};
    int matrix[n][n];
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int k;
            std::cin >> k;
            matrix[i][j] = k;
            sumRow[i] = sumRow[i] + k;
            sumCol[j] = sumCol[j] + k;
            if (i == j) {
                sumDiag[0] = sumDiag[0] + k;
            }
            if (i == (n-j-1)) {
                sumDiag[1] = sumDiag[1] + k;
            }
        }
    }
    
    int SUM = sumDiag[0];
    bool status = true;
    
    if (sumDiag[1] != SUM) {status = false;}
    for (int i = 0; i < n; i++) {
        if (SUM != sumCol[i]) {
            std::cout << sumCol[i];
            status = false;
        }
        if (SUM != sumRow[i]) {
            std::cout << sumRow[i];
            status = false;
        }
    }
    if (status == true) {
        std::cout << "MAGICCCCC!!!";
    } else {
        std::cout << "not mag.. 😭😭😭😭!!!";
    }
}

void Problem9() {
    int b = 6;
    int* a = &b;
    
    b = 7;
    std::cout << *a << "\n";
}

#include <CoreFoundation/CoreFoundation.h>
#include <QuartzCore/QuartzCore.hpp>
#include <ApplicationServices/ApplicationServices.h>

static void eventLoop(CFRunLoopRef runLoop) {
    // A basic event loop to handle events like quit
    while (true) {
        CFRunLoopRun();
    }
}


//#ifdef __cplusplus
//extern "C" {
//#endif





//#include <Cocoa/Cocoa.h>

// Place C-compatible declarations here
int myCFunction() {
    size_t width = 800;
    size_t height = 600;

    // Create a color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (!colorSpace) {
        fprintf(stderr, "Failed to create color space!\n");
        return -1;
    }

    // Create a bitmap context
    CGContextRef context = CGBitmapContextCreate(
        NULL,                     // Data buffer (NULL for CoreGraphics to allocate)
        width,                    // Width
        height,                   // Height
        8,                        // Bits per component
        width * 4,                // Bytes per row (RGBA, 4 bytes per pixel)
        colorSpace,               // Color space
        kCGImageAlphaPremultipliedLast // Bitmap info
    );

    if (!context) {
        fprintf(stderr, "Failed to create bitmap context!\n");
        CGColorSpaceRelease(colorSpace);
        return -1;
    }

    // Set fill color to green
    CGContextSetRGBFillColor(context, 0.0, 1.0, 0.0, 1.0); // Green color

    // Fill a rectangle
    CGContextFillRect(context, CGRectMake(100, 100, 400, 300));

    // Create a CGImage from the context
    CGImageRef image = CGBitmapContextCreateImage(context);
    if (!image) {
        fprintf(stderr, "Failed to create image from context!\n");
        CGContextRelease(context);
        CGColorSpaceRelease(colorSpace);
        return -1;
    }

    // Save the image to a file (PNG format)
    CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, CFSTR("/Users/adityadude/Desktop/output1.png"), kCFURLPOSIXPathStyle, false);
    CGImageDestinationRef destination = CGImageDestinationCreateWithURL(url, kUTTypePNG, 1, NULL);
    if (destination) {
        CGImageDestinationAddImage(destination, image, NULL);
        if (!CGImageDestinationFinalize(destination)) {
            fprintf(stderr, "Failed to write image to file!\n");
        }
        CFRelease(destination);
    } else {
        fprintf(stderr, "Failed to create image destination!\n");
    }

    // Clean up
//    CFRelease(url);
    CGImageRelease(image);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);

    printf("Drawing complete. Check the 'output.png' file.\n");

    return 0;

} // Example function



int myWindowFunc() {
    
    // Initialize the application services
    CFRunLoopRef runLoop = CFRunLoopGetCurrent();
    
    // Set up a basic window using Quartz Event Services
    // Create the window properties
    CGRect windowRect = CGRectMake(100, 100, 800, 600);  // Define window size and position
    CFStringRef windowTitle = CFSTR("Quartz Event Services Window");

    // Create the window using Quartz
    CGEventMask eventMask = kCGEventMaskForAllEvents;
    CGEventSourceRef eventSource = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
    
    CGEventRef event = CGEventCreate(NULL);
//    CGEventPost(kCGEventSourceStateHIDSystemState, event);
    CFRelease(eventSource);
    
    // This is a minimal event loop (basic skeleton) that will keep the window alive
    eventLoop(runLoop);

    return 0;
}

//#ifdef __cplusplus
//}
//#endif

//int Carrrr() {
//    // Create a simple Carbon application with a window
//    CFStringRef appName = CFStringCreateWithCString(kCFAllocatorDefault, "CarbonApp", kCFStringEncodingUTF8);
//    if (appName == nullptr) {
//        std::cerr << "Failed to create application name!" << std::endl;
//        return -1;
//    }
//
//    // Create a simple window using Carbon API (example)
//    WindowRef window;
//    Rect windowRect = {100, 100, 400, 400};
//    OSStatus err = CreateNewWindow(kDocumentWindowClass, kWindowStandardHandlerAttribute, &windowRect, &window);
//    if (err != noErr) {
//        std::cerr << "Failed to create window!" << std::endl;
//        CFRelease(appName);
//        return -1;
//    }
//
//    // Display the window
//    ShowWindow(window);
//
//    // Start the event loop
//    EventLoop();
//    
//    // Clean up
//    CFRelease(appName);
//
//    return 0;
//}

extern "C" int NSRunAlertPanel(CFStringRef strTitle, CFStringRef strMsg,
                               CFStringRef strButton1, CFStringRef strButton2,
                               CFStringRef strButton3, ...);


int mainWW(int argc, char** argv)
{
    id app = NULL;
    id pool = (id)objc_getClass("NSAutoreleasePool");
    if (!pool)
    {
        std::cerr << "Unable to get NSAutoreleasePool!\nAborting\n";
        return -1;
    }
    pool = objc_msgSend(pool, sel_registerName("alloc"));
    if (!pool)
    {
        std::cerr << "Unable to create NSAutoreleasePool...\nAborting...\n";
        return -1;
    }
    pool = objc_msgSend(pool, sel_registerName("init"));

    app = objc_msgSend((id)objc_getClass("NSApplication"),
                       sel_registerName("sharedApplication"));

    NSRunAlertPanel(CFSTR("Testing"),
                    CFSTR("This is a simple test to display NSAlertPanel."),
                    CFSTR("OK"), NULL, NULL);

    objc_msgSend(pool, sel_registerName("release"));
    return 0;
}




int main(int argc, const char * argv[]) {
    objectiveCPlusPlusFunction();
//    fibnac(7);
    LOFFFF();
//    myWindowFunc();
    // Initialize the application services
//    CFRunLoopRef runLoop = CFRunLoopGetCurrent();
//    
//    // Set up a basic window using Quartz Event Services
//    // Create the window properties
//    CGRect windowRect = CGRectMake(100, 100, 800, 600);  // Define window size and position
//    CFStringRef windowTitle = CFSTR("Quartz Event Services Window");
//
//    // Create the window using Quartz
//    CGEventMask eventMask = kCGEventMaskForAllEvents;
//    CGEventSourceRef eventSource = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
//    
//    CGEventRef event = CGEventCreate(NULL);
////    CGEventPost(kCGEventSourceStateHIDSystemState, event);
//    CFRelease(eventSource);
//    
//    // This is a minimal event loop (basic skeleton) that will keep the window alive
//    eventLoop(runLoop);

    return 0;
}
