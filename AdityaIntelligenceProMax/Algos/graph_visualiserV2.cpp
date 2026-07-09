//
// Created by Aditya Dudeja on 07/06/26.
//

#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <filesystem>
#include <cstdlib>

#include "../matrix.h"
#include "../primitives.cpp"

struct MatrixIdentity {
    void* buffer;
    const Primitive* tape;

    bool operator==(const MatrixIdentity& other) const {
        return buffer == other.buffer && tape == other.tape;
    }
};

struct MatrixIdentityHash {
    std::size_t operator()(const MatrixIdentity& id) const {
        return std::hash<void*>()(id.buffer) ^ (std::hash<const Primitive*>()(id.tape) << 1);
    }
};

class GraphVisualiserV2 {
    std::unordered_map<MatrixIdentity, std::string, MatrixIdentityHash> matrix_names;
    int matrix_name_counter = 0;
    int cluster_counter = 0;

    std::unordered_set<const Primitive*> visited_prims;
    std::unordered_set<MatrixIdentity, MatrixIdentityHash> visited_mats;

    std::string get_matrix_name(const matrix& m) {
        MatrixIdentity id = {m.buffer, m.tape};
        if (matrix_names.find(id) == matrix_names.end()) {
            int c = matrix_name_counter++;
            std::string name = "";
            do {
                name = (char)('A' + (c % 26)) + name;
                c = c / 26 - 1;
            } while (c >= 0);
            matrix_names[id] = name;
        }
        return matrix_names[id];
    }

    std::string mat_node_id(const matrix& m) {
        return "M_" + get_matrix_name(m);
    }

    std::string prim_node_id(const Primitive* p) {
        return "P_" + std::to_string(reinterpret_cast<uintptr_t>(p));
    }

    const char* prim_label(const Primitive* p) {
        if (dynamic_cast<const SwapLeafPrimitive*>(p))       return "Leaf";
        if (dynamic_cast<const AdditionPrimitive*>(p))       return "+";
        if (dynamic_cast<const SubtractionPrimitive*>(p))    return "−";
        if (dynamic_cast<const MultiplicationPrimitive*>(p)) return "×";
        if (dynamic_cast<const DivisionPrimitive*>(p))       return "÷";
        if (dynamic_cast<const StackPrimitive*>(p))          return "Stack";
        if (dynamic_cast<const SlicePrimitive*>(p))          return "Slice";
        if (dynamic_cast<const SumPrimitive*>(p))            return "Sum";
        if (dynamic_cast<const PaddingPrimitive*>(p))        return "Padding";
        if (dynamic_cast<const BrodcastPrimitive*>(p))       return "Broadcast";
        if (dynamic_cast<const ReshapePrimitive*>(p))        return "Reshape";
        if (dynamic_cast<const AsTypePrimitive*>(p))         return "AsType";
        if (dynamic_cast<const TransposePrimitive*>(p))      return "Transpose";
        if (dynamic_cast<const CompiledNodePrimitive*>(p))   return "Compiled";
        return "?";
    }

    void traverse(const matrix& m, std::ofstream& dot) {
        MatrixIdentity id = {m.buffer, m.tape};
        if (visited_mats.count(id)) return;
        visited_mats.insert(id);

        std::string m_name = get_matrix_name(m);
        std::string m_id = mat_node_id(m);

        dot << "  " << m_id << " [shape=ellipse, label=\"" << m_name << "\", style=filled, fillcolor=\"#d4f1e5\"];\n";

        if (!m.tape) return;
        const Primitive* p = m.tape;

        if (!visited_prims.count(p)) {
            visited_prims.insert(p);
            std::string p_id = prim_node_id(p);
            dot << "  " << p_id << " [shape=box, label=\"" << prim_label(p) << "\", style=filled, fillcolor=\"#cbc8f5\"];\n";

            if (auto* ap = dynamic_cast<const AdditionPrimitive*>(p)) {
                traverse(ap->a, dot); traverse(ap->b, dot);
                dot << "  " << mat_node_id(ap->a) << " -> " << p_id << " [label=\"a\"];\n";
                dot << "  " << mat_node_id(ap->b) << " -> " << p_id << " [label=\"b\"];\n";
            }
            else if (auto* sp = dynamic_cast<const SubtractionPrimitive*>(p)) {
                traverse(sp->a, dot); traverse(sp->b, dot);
                dot << "  " << mat_node_id(sp->a) << " -> " << p_id << " [label=\"a\"];\n";
                dot << "  " << mat_node_id(sp->b) << " -> " << p_id << " [label=\"b\"];\n";
            }
            else if (auto* mp = dynamic_cast<const MultiplicationPrimitive*>(p)) {
                traverse(mp->a, dot); traverse(mp->b, dot);
                dot << "  " << mat_node_id(mp->a) << " -> " << p_id << " [label=\"a\"];\n";
                dot << "  " << mat_node_id(mp->b) << " -> " << p_id << " [label=\"b\"];\n";
            }
            else if (auto* dp = dynamic_cast<const DivisionPrimitive*>(p)) {
                traverse(dp->a, dot); traverse(dp->b, dot);
                dot << "  " << mat_node_id(dp->a) << " -> " << p_id << " [label=\"a\"];\n";
                dot << "  " << mat_node_id(dp->b) << " -> " << p_id << " [label=\"b\"];\n";
            }
            else if (auto* stack = dynamic_cast<const StackPrimitive*>(p)) {
                for (size_t i = 0; i < stack->inputs.size(); ++i) {
                    traverse(stack->inputs[i], dot);
                    dot << "  " << mat_node_id(stack->inputs[i]) << " -> " << p_id << " [label=\"i" << i << "\"];\n";
                }
            }
            else if (auto* sl = dynamic_cast<const SlicePrimitive*>(p)) {
                traverse(sl->input, dot);
                dot << "  " << mat_node_id(sl->input) << " -> " << p_id << ";\n";
            }
            else if (auto* sum = dynamic_cast<const SumPrimitive*>(p)) {
                traverse(sum->input, dot);
                dot << "  " << mat_node_id(sum->input) << " -> " << p_id << ";\n";
            }
            else if (auto* pad = dynamic_cast<const PaddingPrimitive*>(p)) {
                traverse(pad->input, dot); traverse(pad->value, dot);
                dot << "  " << mat_node_id(pad->input) << " -> " << p_id << " [label=\"in\"];\n";
                dot << "  " << mat_node_id(pad->value) << " -> " << p_id << " [label=\"val\"];\n";
            }
            else if (auto* br = dynamic_cast<const BrodcastPrimitive*>(p)) {
                traverse(br->input, dot);
                dot << "  " << mat_node_id(br->input) << " -> " << p_id << ";\n";
            }
            else if (auto* re = dynamic_cast<const ReshapePrimitive*>(p)) {
                traverse(re->input, dot);
                dot << "  " << mat_node_id(re->input) << " -> " << p_id << ";\n";
            }
            else if (auto* as = dynamic_cast<const AsTypePrimitive*>(p)) {
                traverse(as->input, dot);
                dot << "  " << mat_node_id(as->input) << " -> " << p_id << ";\n";
            }
            else if (auto* tr = dynamic_cast<const TransposePrimitive*>(p)) {
                traverse(tr->input, dot);
                dot << "  " << mat_node_id(tr->input) << " -> " << p_id << ";\n";
            }
            else if (auto* cp = dynamic_cast<const CompiledNodePrimitive*>(p)) {
                traverse(cp->outer_input, dot);
                traverse(cp->sample_parameter, dot);
                traverse(cp->output_graph, dot);
                dot << "  " << mat_node_id(cp->outer_input) << " -> " << p_id << " [label=\"out_in\"];\n";
                dot << "  " << mat_node_id(cp->sample_parameter) << " -> " << p_id << " [label=\"sample\"];\n";
                dot << "  " << mat_node_id(cp->output_graph) << " -> " << p_id << " [label=\"out_g\"];\n";
            }
            else if (auto* sw = dynamic_cast<const SwapLeafPrimitive*>(p)) {
                // leaf node, no incoming edges from other primitives
            }
        }

        std::string p_id = prim_node_id(p);
        dot << "  " << p_id << " -> " << m_id << ";\n";
    }

public:
    void visualise(const matrix& output, const std::string& path = "") {
        std::string outpath = path.empty()
            ? std::string("/Users/") + getenv("USER") + "/Documents/output_images/graph_v2.dot"
            : path;

        std::filesystem::create_directories(std::filesystem::path(outpath).parent_path());

        std::ofstream dot(outpath);
        dot << "digraph ComputeGraph {\n";
        dot << "  rankdir=LR;\n";
        dot << "  node [fontname=\"Helvetica\", fontsize=12];\n";

        traverse(output, dot);

        dot << "}\n";
        dot.flush();
        dot.close();

        std::cout << "Graph written to: " << outpath << std::endl;

        std::string cmd = "/opt/homebrew/bin/dot -Tpng " + outpath + " -o " + outpath + ".png && open " + outpath + ".png";
        
#if TARGET_OS_IPHONE
    // You cannot use std::system here.
    // If you need to make a directory, download a file, or unzip something,
    // you must use Apple's native APIs (like NSFileManager).
    NSLog(@"Shell commands are blocked on iOS!");
#else
    // On macOS, you have a terminal, so this works perfectly!
    std::system(cmd.c_str());
#endif
    }
};

void visualise_graph_v2(const matrix& output, const std::string& path = "") {
    GraphVisualiserV2 visualiser;
    visualiser.visualise(output, path);
}
