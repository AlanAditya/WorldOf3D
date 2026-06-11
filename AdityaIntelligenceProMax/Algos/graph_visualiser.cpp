//
// Created by Aditya Dudeja on 07/06/26.
//

#include <fstream>
#include <unordered_set>
#include "../matrix.h"
#include "../primitives.cpp"


// Forward declare so Primitive can call it
static void traverse(const matrix& m, std::ofstream& dot,
                     std::unordered_set<const Primitive*>& visited,
                     std::unordered_set<const matrix*>& mat_visited);

static std::string mat_name(const matrix* m) {
    return "M" + std::to_string(reinterpret_cast<uintptr_t>(m));
}

static std::string prim_name(const Primitive* p) {
    return "P" + std::to_string(reinterpret_cast<uintptr_t>(p));
}

static const char* prim_label(const Primitive* p) {
    if (dynamic_cast<const AdditionPrimitive*>(p))       return "+";
    if (dynamic_cast<const SubtractionPrimitive*>(p))    return "−";
    if (dynamic_cast<const MultiplicationPrimitive*>(p)) return "×";
    if (dynamic_cast<const DivisionPrimitive*>(p))       return "÷";
    return "?";
}

static void traverse(const matrix& m, std::ofstream& dot,
                     std::unordered_set<const Primitive*>& visited,
                     std::unordered_set<const matrix*>& mat_visited) {
    if (mat_visited.count(&m)) return;
    mat_visited.insert(&m);

    dot << "  " << mat_name(&m)
        << " [shape=ellipse, label=\"M\", style=filled, fillcolor=\"#d4f1e5\"];\n";

    if (!m.tape) return;
    const Primitive* p = m.tape;

    if (!visited.count(p)) {
        visited.insert(p);
        dot << "  " << prim_name(p)
            << " [shape=box, label=\"" << prim_label(p)
            << "\", style=filled, fillcolor=\"#cbc8f5\"];\n";

        // recurse into inputs
        if (auto* ap = dynamic_cast<const AdditionPrimitive*>(p)) {
            traverse(ap->a, dot, visited, mat_visited);
            traverse(ap->b, dot, visited, mat_visited);
            dot << "  " << mat_name(&ap->a) << " -> " << prim_name(p) << " [label=\"a\"];\n";
            dot << "  " << mat_name(&ap->b) << " -> " << prim_name(p) << " [label=\"b\"];\n";
        } else if (auto* sp = dynamic_cast<const SubtractionPrimitive*>(p)) {
            traverse(sp->a, dot, visited, mat_visited);
            traverse(sp->b, dot, visited, mat_visited);
            dot << "  " << mat_name(&sp->a) << " -> " << prim_name(p) << " [label=\"a\"];\n";
            dot << "  " << mat_name(&sp->b) << " -> " << prim_name(p) << " [label=\"b\"];\n";
        } else if (auto* mp = dynamic_cast<const MultiplicationPrimitive*>(p)) {
            traverse(mp->a, dot, visited, mat_visited);
            traverse(mp->b, dot, visited, mat_visited);
            dot << "  " << mat_name(&mp->a) << " -> " << prim_name(p) << " [label=\"a\"];\n";
            dot << "  " << mat_name(&mp->b) << " -> " << prim_name(p) << " [label=\"b\"];\n";
        } else if (auto* dp = dynamic_cast<const DivisionPrimitive*>(p)) {
            traverse(dp->a, dot, visited, mat_visited);
            traverse(dp->b, dot, visited, mat_visited);
            dot << "  " << mat_name(&dp->a) << " -> " << prim_name(p) << " [label=\"a\"];\n";
            dot << "  " << mat_name(&dp->b) << " -> " << prim_name(p) << " [label=\"b\"];\n";
        }
    }

    dot << "  " << prim_name(p) << " -> " << mat_name(&m) << ";\n";
}

void visualise_graph(const matrix& output, const std::string& path = "") {
    std::string outpath = path.empty()
        ? std::string("/Users/") + getenv("USER") + "/Documents/output_images/graph.dot"
        : path;
    
    std::filesystem::create_directories(std::filesystem::path(outpath).parent_path());
    
    std::ofstream dot(outpath);
    dot << "digraph ComputeGraph {\n";
    dot << "  rankdir=LR;\n";
    dot << "  node [fontname=\"Helvetica\", fontsize=12];\n";

    std::unordered_set<const Primitive*> visited;
    std::unordered_set<const matrix*> mat_visited;
    traverse(output, dot, visited, mat_visited);

    dot << "}\n";
    dot.flush();
    dot.close();

    std::cout << "Graph written to: " << outpath << std::endl;

    std::string cmd = "/opt/homebrew/bin/dot -Tpng " + outpath + " -o " + outpath + ".png && open " + outpath + ".png";
    std::system(cmd.c_str());
}
