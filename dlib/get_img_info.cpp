#include "get_img_info.hpp"

#include <algorithm>

#include <dlib/xml_parser.h>

namespace ocv{


namespace odlib{

class doc_handler : public dlib::document_handler
{
public:
    doc_handler(std::vector<std::string> &img_name,
                std::vector<std::vector<dlib::rectangle>> &location) :
        img_name_(img_name),
        location_(location)
    {

    }

    virtual void start_document () override
    {
    }

    virtual void end_document () override
    {
    }

    virtual void start_element (
            const unsigned long,
            const std::string& name,
            const dlib::attribute_list& atts) override
    {
        if(name == "image"){
            atts.reset();
            if(atts.move_next()){
                img_name_.emplace_back(atts.element().value());
                location_.emplace_back();
            }
        }else if(name == "box"){
            atts.reset();
            dlib::rectangle rect;
            long h = 0;
            enum {height, left, top, width};
            for(size_t index = 0; atts.move_next(); ++index){
                switch(index){
                case height :{
                    h = std::stol(atts.element().value()) - 1;
                    h = std::max<long>(0, h);
                    break;
                }
                case left :{
                    rect.set_left(std::stol(atts.element().value()));
                    break;
                }
                case top :{
                    rect.set_top(std::stol(atts.element().value()));
                    break;
                }
                case width:{
                    rect.set_right(rect.left() + std::stol(atts.element().value()) - 1);
                    rect.set_right(std::max<long>(0, rect.right()));
                    break;
                }
                default:{
                    break;
                }
                }
            }
            rect.set_bottom(rect.top() + h);
            location_.back().emplace_back(rect);
        }
    }

    virtual void end_element (
            const unsigned long,
            const std::string&
            ) override
    {
    }

    virtual void characters (
            const std::string&
            ) override
    {
    }

    virtual void processing_instruction (const unsigned long, const std::string&,
                                         const std::string&
                                         ) override
    {
    }

private:
    std::vector<std::string> &img_name_;
    std::vector<std::vector<dlib::rectangle>> &location_;
};

void get_imglab_xml_info(std::vector<std::string> &img_name,
                    std::vector<std::vector<dlib::rectangle>> &roi,
                    const std::string &file_name)
{
    doc_handler handler(img_name, roi);
    dlib::parse_xml(file_name, handler);
}

}

}
