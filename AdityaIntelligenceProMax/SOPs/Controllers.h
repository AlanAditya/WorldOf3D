#pragma once
#include <string>
#include <vector>
#include "matrix.h"
#include "GeoNode.cpp"
#include "MeshPrimitives.h"

class PointCloudController {
private:
    matrix x_stream;
    matrix y_stream;

public:
    GeoNodeImpl node;

    PointCloudController(std::string name, size_t point_count) 
        : x_stream(0, dtype::Float), y_stream(0, dtype::Float) 
    {
        node = GeoNode::create(std::move(name));
        
        // Setup initial graph data
        x_stream = matrix::zeros({(size_m)point_count, 1}, dtype::Float);
        y_stream = matrix::zeros({(size_m)point_count, 1}, dtype::Float);
        matrix z_stream = matrix::zeros({(size_m)point_count, 1}, dtype::Float);

        // Pack handles into the generic node payload
        std::vector<matrix> streams = {x_stream, y_stream, z_stream};
        node->mesh.vert_position = matrix::concat(streams, 1);
        
        matrix indices(1, dtype::UInt32);
        indices.total_size = point_count;
        indices.shape()[0] = point_count;
        indices.calcStrides();
        indices.buffer = new uint8_t[point_count * sizeof(uint32_t)];
        uint32_t* buf = (uint32_t*)indices.buffer;
        for (size_t i = 0; i < point_count; ++i) buf[i] = (uint32_t)i;
        if (point_count > 10) indices.buildMetalBuffer();
        
        node->mesh.indices = indices;
    }

    void update(const matrix& new_x, const matrix& new_y) {
        this->x_stream = new_x;
        this->y_stream = new_y;
        
        matrix z_stream = matrix::zeros({(size_m)x_stream.shape()[0], 1}, dtype::Float);
        std::vector<matrix> streams = {x_stream, y_stream, z_stream};
        node->mesh.vert_position = matrix::concat(streams, 1);
    }
};

class TriangleController {
public:
    GeoNodeImpl node;

    TriangleController(std::string name, matrix anchor = matrix(0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::triangle(1.0f, 1.0f);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class QuadController {
public:
    GeoNodeImpl node;

    QuadController(std::string name, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::quad(1.0f, 1.0f);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class CubeController {
public:
    GeoNodeImpl node;

    CubeController(std::string name, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::cube(1.0f, 1.0f, 1.0f);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class CircleController {
public:
    GeoNodeImpl node;

    CircleController(std::string name, matrix anchor = matrix(0, 0,dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::circle(1.0f, 32);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class SphereController {
public:
    GeoNodeImpl node;

    SphereController(std::string name, matrix anchor = matrix(0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::sphere(1.0f, 16, 16);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};
