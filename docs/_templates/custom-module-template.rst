{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
    :members:

    {% block modules %}
    {% if modules %}
    .. rubric:: {{ _('Sub-Modules') }}
    .. autosummary::
       :toctree:
       :template: custom-module-template.rst
       :recursive:
    {% for item in modules %}
       {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Module Attributes') }}
    .. autosummary::
      :toctree:
    {% for item in attributes %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Module Functions') }}
    .. autosummary::
      :toctree:
      :nosignatures:
    {% for item in functions %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Module Classes') }}
    .. autosummary::
      :toctree:
      :template: custom-class-template.rst
      :nosignatures:
    {% for item in classes %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: {{ _('Module Exceptions') }}
    .. autosummary::
      :toctree:
    {% for item in exceptions %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

