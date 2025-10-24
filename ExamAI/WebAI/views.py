from django.shortcuts import render
from django.views import View

# Create your views here.
class MainPageView(View):

    def get(self, req):
        template_name = 'WebAI/main.html'
        context = {

        }
        return render(req, template_name, context=context)


class ComputerVisionPageView(View):

    def get(self, req):
        template_name = 'WebAI/computer_vision.html'
        context = {

        }
        return render(req, template_name, context=context)